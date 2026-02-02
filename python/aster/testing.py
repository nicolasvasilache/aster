"""Test utilities for aster kernels.

This module provides common utilities for compiling and running AMDGCN kernels in tests.
Consolidates functionality from integration_test/test_utils.py and
mlir_kernels/test/test_utils.py.
"""

import os
import logging
import numpy as np
from contextlib import contextmanager
from typing import Tuple, Callable, Optional, List, Any, Generator

from aster import ir, utils
from aster.dialects import amdgcn
from aster.pass_pipelines import (
    DEFAULT_SROA_PASS_PIPELINE,
    TEST_SYNCHRONOUS_SROA_PASS_PIPELINE,
)
from aster._mlir_libs._runtime_module import (
    hip_init,
    hip_module_load_data,
    hip_module_get_function,
    hip_module_launch_kernel,
    hip_device_synchronize,
    hip_free,
    hip_malloc,
    hip_memcpy_host_to_device,
    hip_memcpy_device_to_host,
    hip_module_unload,
    hip_function_free,
    hip_get_device_count,
    hip_set_device,
    hip_get_device,
    hip_event_create,
    hip_event_destroy,
    hip_event_record,
    hip_event_synchronize,
    hip_event_elapsed_time,
)

# Default test configuration
DEFAULT_MCPU = "gfx942"
DEFAULT_WAVEFRONT_SIZE = 64

__all__ = [
    # Constants
    "DEFAULT_MCPU",
    "DEFAULT_WAVEFRONT_SIZE",
    # Logging
    "MillisecondFormatter",
    "_get_logger",
    "_should_log",
    "_log_info",
    "_log_with_device",
    # Context managers
    "hsaco_file",
    # Compilation
    "load_mlir_module_from_file",
    "compile_mlir_file_to_asm",
    # Execution
    "execute_kernel_and_verify",
    "compile_and_run",
]


# =============================================================================
# Logging
# =============================================================================


class MillisecondFormatter(logging.Formatter):
    """Formatter that includes milliseconds in timestamps."""

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        msecs = int(record.msecs)
        return f"{ct.tm_hour:02d}:{ct.tm_min:02d}:{ct.tm_sec:02d}.{msecs:03d}"


def _get_logger():
    """Get logger configured for multiprocessing-safe logging."""
    logger = logging.getLogger("aster.testing")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            MillisecondFormatter(fmt="%(asctime)s [PID:%(process)d] %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _should_log():
    """Check if logging is enabled via ASTER_LOGGING environment variable."""
    return bool(os.getenv("ASTER_LOGGING"))


def _log_info(logger, message):
    """Log info message if logging is enabled."""
    if not _should_log():
        return
    logger.info(message)


def _log_with_device(logger, device_id, message):
    """Log message with device_id."""
    if not _should_log():
        return
    device_str = f"GPU{device_id}" if device_id is not None else "GPU?"
    logger.info(f"[{device_str}] {message}")


# =============================================================================
# Context managers
# =============================================================================


@contextmanager
def hsaco_file(path: str) -> Generator[str, None, None]:
    """Context manager that cleans up an HSACO file on exit."""
    try:
        yield path
    finally:
        if path and os.path.exists(path):
            os.unlink(path)


# =============================================================================
# MLIR compilation
# =============================================================================


def load_mlir_module_from_file(
    file_path: str, ctx, preprocess: Optional[Callable[[str], str]] = None
):
    """Load MLIR module from file."""
    from aster._mlir_libs._mlir import ir as mlir_ir

    with open(file_path, "r") as f:
        mlir_content = f.read()

    if preprocess is not None:
        mlir_content = preprocess(mlir_content)

    ctx.allow_unregistered_dialects = True
    with mlir_ir.Location.unknown():
        module = mlir_ir.Module.parse(mlir_content, context=ctx)
    return module


def compile_mlir_file_to_asm(
    mlir_file: str,
    kernel_name: str,
    pass_pipeline: str,
    ctx,
    preprocess: Optional[Callable[[str], str]] = None,
    print_ir_after_all: bool = False,
    library_paths: Optional[List[str]] = None,
    print_timings: bool = False,
) -> Tuple[str, Any]:
    """Compile MLIR file to assembly.

    Returns:
        Tuple of (asm_code, module)
    """
    logger = _get_logger()
    _log_info(logger, f"[COMPILE] Loading MLIR file: {os.path.basename(mlir_file)}")

    module = load_mlir_module_from_file(mlir_file, ctx, preprocess)

    from aster._mlir_libs._mlir import passmanager

    # Pre-apply preload-library pass if library paths are provided
    if library_paths:
        for lib_path in library_paths:
            if not os.path.exists(lib_path):
                raise FileNotFoundError(
                    f"Library file not found: {lib_path}. MLIR file: {mlir_file}"
                )
        _log_info(logger, "[COMPILE] Pre-applying preload-library pass")
        paths_str = ",".join(library_paths)
        preload_pass = (
            f"builtin.module(amdgcn-preload-library{{library-paths={paths_str}}})"
        )
        pm = passmanager.PassManager.parse(preload_pass, ctx)
        pm.run(module.operation)

    _log_info(logger, "[COMPILE] Applying pass pipeline")
    pm = passmanager.PassManager.parse(pass_pipeline, ctx)
    if print_ir_after_all:
        pm.enable_ir_printing()
    if print_timings:
        pm.enable_timing()
    pm.run(module.operation)
    _log_info(logger, "[COMPILE] Pass pipeline completed")

    # Find the amdgcn.kernel inside the proper amdgcn.module
    _log_info(logger, f"[COMPILE] Searching for kernel: {kernel_name}")
    amdgcn_module = None
    found_kernel = False
    for op in module.body:
        if not isinstance(op, amdgcn.ModuleOp):
            continue
        amdgcn_module = op
        for kernel_op in amdgcn_module.body_region.blocks[0].operations:
            if not isinstance(kernel_op, amdgcn.KernelOp):
                continue
            if kernel_op.sym_name.value == kernel_name:
                found_kernel = True
                break
        if found_kernel:
            break

    assert amdgcn_module is not None, "Failed to find any AMDGCN module"
    assert found_kernel, f"Failed to find kernel {kernel_name}"
    _log_info(logger, f"[COMPILE] Found kernel: {kernel_name}")

    _log_info(logger, "[COMPILE] Translating to assembly")
    asm_complete = utils.translate_module(amdgcn_module, debug_print=False)
    _log_info(logger, "[COMPILE] Assembly generation completed")

    return asm_complete, module


# =============================================================================
# Kernel execution
# =============================================================================


def execute_kernel_and_verify(
    hsaco_path: Optional[str],
    kernel_name: str,
    input_args: List[np.ndarray],
    output_args: List[np.ndarray],
    mcpu: str,
    wavefront_size: int = 64,
    grid_dim: Tuple[int, int, int] = (1, 1, 1),
    block_dim: Tuple[int, int, int] = (64, 1, 1),
    verify_fn: Optional[Callable[[List[np.ndarray], List[np.ndarray]], None]] = None,
    padding_bytes: Optional[List[int]] = None,
    num_iterations: int = 1,
    device_id: Optional[int] = None,
    flush_llc: Optional[Any] = None,
) -> List[int]:
    """Execute a GPU kernel and verify its results.

    Returns:
        List of execution times in nanoseconds, one per iteration
    """
    assert all(
        array.size > 0 for array in input_args + output_args
    ), "All NP arrays must have > 0 elements"

    if hsaco_path is None:
        raise RuntimeError("Failed to assemble kernel to HSACO")

    logger = _get_logger()
    hip_init()
    if device_id is not None:
        hip_set_device(device_id)

    gpu_ptrs: Optional[List[Any]] = None
    padded_buffers: Optional[List[Any]] = None
    has_padding = False
    module = None
    function = None
    start_event = None
    stop_event = None

    actual_device_id = device_id if device_id is not None else hip_get_device()

    try:
        _log_with_device(
            logger,
            actual_device_id,
            f"Starting execution: kernel={kernel_name}, iterations={num_iterations}",
        )

        with open(hsaco_path, "rb") as f:
            hsaco_binary = f.read()

        module = hip_module_load_data(hsaco_binary)
        function = hip_module_get_function(module, kernel_name.encode())
        _log_with_device(
            logger, actual_device_id, f"Loaded HSACO: {os.path.basename(hsaco_path)}"
        )

        all_arrays = input_args + output_args

        if padding_bytes is None:
            padding_bytes = [0] * len(all_arrays)
        elif len(padding_bytes) != len(all_arrays):
            raise ValueError(
                f"padding_bytes must have {len(all_arrays)} elements, got {len(padding_bytes)}"
            )

        has_padding = any(pb > 0 for pb in padding_bytes)

        if has_padding:
            padded_buffers = []
            ptr_values = []
            _log_with_device(
                logger, actual_device_id, f"Allocating {len(all_arrays)} padded buffers"
            )
            for arr, pad_bytes in zip(all_arrays, padding_bytes):
                base_ptr, data_ptr, _ = utils.copy_array_to_gpu(arr, pad_bytes)
                padded_buffers.append(base_ptr)
                ptr_values.append(utils.unwrap_pointer_from_capsule(data_ptr))
            params_tuple = utils.create_kernel_args_capsule(ptr_values)
        else:
            _log_with_device(
                logger, actual_device_id, f"Allocating {len(all_arrays)} buffers"
            )
            params_tuple, gpu_ptrs = utils.create_kernel_args_capsule_from_numpy(
                *all_arrays, device_id=device_id
            )

        iteration_times_ns = []
        start_event = hip_event_create()
        stop_event = hip_event_create()

        _log_with_device(
            logger,
            actual_device_id,
            f"Launching kernel: grid={grid_dim}, block={block_dim}",
        )

        if flush_llc is not None:
            flush_llc.initialize()

        for iteration in range(num_iterations):
            if flush_llc is not None:
                flush_llc.flush_llc()

            hip_event_record(start_event)
            hip_module_launch_kernel(
                function,
                grid_dim[0],
                grid_dim[1],
                grid_dim[2],
                block_dim[0],
                block_dim[1],
                block_dim[2],
                params_tuple[0],
            )
            hip_event_record(stop_event)
            hip_event_synchronize(stop_event)

            elapsed_ms = hip_event_elapsed_time(start_event, stop_event)
            elapsed_ns = int(elapsed_ms * 1_000_000)
            iteration_times_ns.append(elapsed_ns)

            _log_with_device(
                logger, actual_device_id, f"Iteration {iteration}: {elapsed_ms:.3f}ms"
            )

            if iteration == 0:
                _log_with_device(logger, actual_device_id, "Verifying results")
                num_inputs = len(input_args)
                if has_padding:
                    assert padded_buffers is not None
                    for i, output_arr in enumerate(output_args):
                        utils.copy_from_gpu_buffer(
                            padded_buffers[num_inputs + i],
                            output_arr,
                            padding_bytes[num_inputs + i],
                        )
                else:
                    assert gpu_ptrs is not None
                    for i, output_arr in enumerate(output_args):
                        capsule_output = utils.wrap_pointer_in_capsule(
                            output_arr.ctypes.data
                        )
                        hip_memcpy_device_to_host(
                            capsule_output, gpu_ptrs[num_inputs + i], output_arr.nbytes
                        )

                if verify_fn is not None:
                    verify_fn(input_args, output_args)
                    _log_with_device(logger, actual_device_id, "Verification passed")

        avg_time_ms = sum(iteration_times_ns) / len(iteration_times_ns) / 1_000_000
        _log_with_device(
            logger,
            actual_device_id,
            f"Completed {num_iterations} iterations, avg={avg_time_ms:.3f}ms",
        )

        return iteration_times_ns

    finally:
        if start_event is not None:
            hip_event_destroy(start_event)
        if stop_event is not None:
            hip_event_destroy(stop_event)
        if padded_buffers is not None:
            for ptr in padded_buffers:
                hip_free(ptr)
        elif gpu_ptrs is not None:
            for ptr in gpu_ptrs:
                hip_free(ptr)
        if flush_llc is not None:
            flush_llc.cleanup()
        if function is not None:
            hip_function_free(function)
        if module is not None:
            hip_module_unload(module)


# =============================================================================
# High-level test helper
# =============================================================================


def compile_and_run(
    mlir_file: str,
    kernel_name: str,
    input_data: Optional[List[np.ndarray]] = None,
    output_data: Optional[List[np.ndarray]] = None,
    grid_dim: Tuple[int, int, int] = (1, 1, 1),
    block_dim: Tuple[int, int, int] = (64, 1, 1),
    library_paths: Optional[List[str]] = None,
    preprocess: Optional[Callable[[str], str]] = None,
    print_ir_after_all: bool = False,
    pass_pipeline: Optional[str] = None,
    mcpu: str = DEFAULT_MCPU,
    wavefront_size: int = DEFAULT_WAVEFRONT_SIZE,
) -> None:
    """Compile and run a kernel, handling GPU availability checks.

    Args:
        mlir_file: Absolute path to MLIR file
        kernel_name: Name of the kernel to compile and run
        input_data: List of input numpy arrays
        output_data: List of output numpy arrays (modified in-place)
        grid_dim: Grid dimensions for kernel launch
        block_dim: Block dimensions for kernel launch
        library_paths: Optional list of library paths for preload
        preprocess: Optional preprocessing function for MLIR content
        print_ir_after_all: Whether to print IR after all passes
        pass_pipeline: Pass pipeline string (defaults to TEST_SYNCHRONOUS_SROA_PASS_PIPELINE)
        mcpu: Target GPU (default: gfx942)
        wavefront_size: Wavefront size (default: 64)
    """
    import pytest

    if pass_pipeline is None:
        pass_pipeline = TEST_SYNCHRONOUS_SROA_PASS_PIPELINE
    if input_data is None:
        input_data = []
    if output_data is None:
        output_data = []

    with ir.Context() as ctx:
        asm, _ = compile_mlir_file_to_asm(
            mlir_file,
            kernel_name,
            pass_pipeline,
            ctx,
            library_paths=library_paths or [],
            print_ir_after_all=print_ir_after_all,
            preprocess=preprocess,
        )

        hsaco_path = utils.assemble_to_hsaco(
            asm, target=mcpu, wavefront_size=wavefront_size
        )
        if hsaco_path is None:
            raise RuntimeError("Failed to assemble kernel to HSACO")

        with hsaco_file(hsaco_path):
            if not utils.system_has_mcpu(mcpu=mcpu):
                print(asm)
                pytest.skip(f"GPU {mcpu} not available")

            execute_kernel_and_verify(
                hsaco_path=hsaco_path,
                kernel_name=kernel_name,
                input_args=input_data,
                output_args=output_data,
                mcpu=mcpu,
                wavefront_size=wavefront_size,
                grid_dim=grid_dim,
                block_dim=block_dim,
            )
