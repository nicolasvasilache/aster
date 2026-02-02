"""Unit tests for kittens/tiles_16x16.mlir library functions."""

import os
import numpy as np
from typing import List

from aster.testing import compile_and_run
from mlir_kernels.common import get_library_paths

KITTENS_LIBRARY_DIR = os.path.join(os.path.dirname(__file__), "..", "library")


def get_kittens_library_paths() -> List[str]:
    """Get paths to all required library files including kittens."""
    return get_library_paths() + [os.path.join(KITTENS_LIBRARY_DIR, "tiles_16x16.mlir")]


def get_mlir_file(file_name: str) -> str:
    """Get path to a test MLIR file in the kittens test directory."""
    return os.path.join(os.path.dirname(__file__), file_name)


class TestKittensZeroC:
    """Test @zero_C function from kittens/tiles_16x16.mlir."""

    def test_zero_C_produces_zeros(self):
        """Zero-initialized C tile should contain all zeros."""
        output = np.zeros(16 * 16, dtype=np.int32)
        compile_and_run(
            get_mlir_file("test_zero_C.mlir"),
            "test_zero_C",
            output_data=[output],
            library_paths=get_kittens_library_paths(),
        )
        np.testing.assert_array_equal(output, np.zeros(16 * 16, dtype=np.int32))


class TestKittensLoadStoreA:
    """Test @load_A_f16 and @store_A_f16 functions from kittens/tiles_16x16.mlir."""

    def test_load_store_roundtrip(self):
        """Load A tile and store it back - should preserve original data."""
        input_f16 = np.arange(16 * 16, dtype=np.float16)
        input_data = input_f16.view(np.uint16)
        output_data = np.full(16 * 16, 0xFFFF, dtype=np.uint16)
        compile_and_run(
            get_mlir_file("test_load_store_A.mlir"),
            "test_load_store_A",
            input_data=[input_data],
            output_data=[output_data],
            library_paths=get_kittens_library_paths(),
        )
        np.testing.assert_array_equal(output_data, input_data)


class TestKittensMFMA:
    """Test @mfma_f32_16x16x16_f16 function from kittens/tiles_16x16.mlir."""

    def test_mfma_matmul(self):
        """MFMA should compute D = A @ B^T correctly."""
        A = np.eye(16, dtype=np.float16)
        B = np.arange(16 * 16, dtype=np.float16).reshape(16, 16) / 256.0
        D_output = np.zeros(16 * 16, dtype=np.float32)
        compile_and_run(
            get_mlir_file("test_mfma.mlir"),
            "test_mfma",
            input_data=[A.flatten(), B.flatten()],
            output_data=[D_output],
            library_paths=get_kittens_library_paths(),
        )
        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(D_output, expected, rtol=1e-3, atol=1e-3)


class TestKittensGEMM:
    """Test minimal GEMM kernel: C[16x16] = A[16x32] @ B[16x32]^T."""

    def test_gemm_16x16x32(self):
        """GEMM should compute C = A @ B^T correctly with K=32."""
        np.random.seed(42)
        A = (np.random.randn(16, 32) * 0.1).astype(np.float16)
        B = (np.random.randn(16, 32) * 0.1).astype(np.float16)
        C_output = np.zeros(16 * 16, dtype=np.float32)
        compile_and_run(
            get_mlir_file("test_gemm_16x16x32.mlir"),
            "gemm_16x16x32",
            input_data=[A.flatten(), B.flatten()],
            output_data=[C_output],
            library_paths=get_kittens_library_paths(),
        )
        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    TestKittensZeroC().test_zero_C_produces_zeros()
    TestKittensLoadStoreA().test_load_store_roundtrip()
    TestKittensMFMA().test_mfma_matmul()
    TestKittensGEMM().test_gemm_16x16x32()
