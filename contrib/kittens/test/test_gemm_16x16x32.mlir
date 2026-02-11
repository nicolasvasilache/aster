// Minimal kittens GEMM kernel: C = A @ B^T
// A: 16x32 (f16), B: 16x32 (f16), C: 16x16 (f32)
// Uses 2 MFMA iterations (K=16 each) to cover K=32

// Type aliases
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>
!rt_C_f32 = !vx4
!write_token = !amdgcn.write_token<flat>
!future_global_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>

amdgcn.module @kittens_gemm_16x16x32 target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From kittens/tiles_16x16.mlir
  func.func private @load_A_f16(!sx2, index, index, index) -> !future_global_read
  func.func private @load_B_f16(!sx2, index, index, index) -> !future_global_read
  func.func private @zero_C() -> !rt_C_f32
  func.func private @mfma_f32_16x16x16_f16_future(!future_global_read, !future_global_read, !rt_C_f32) -> !rt_C_f32
  func.func private @store_C_f32(!rt_C_f32, !sx2, index, index, index) -> !write_token

  // Minimal GEMM kernel: C[16x16] = A[16x32] @ B[16x32]^T
  // Input: A (16x32 f16, row-major), B (16x32 f16, row-major)
  // Output: C (16x16 f32, row-major)
  amdgcn.kernel @gemm_16x16x32 arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {shared_memory_size = 0 : i32} {
    %A_ptr = amdgcn.load_arg 0 : !sx2
    %B_ptr = amdgcn.load_arg 1 : !sx2
    %C_ptr = amdgcn.load_arg 2 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    // Constants
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index

    // Strides in bytes
    // A and B are 16x32 f16, so stride = 32 * 2 = 64 bytes
    %stride_AB = arith.constant 64 : index  // 32 elements * 2 bytes per f16
    // C is 16x16 f32, so stride = 16 * 4 = 64 bytes
    %stride_C = arith.constant 64 : index   // 16 elements * 4 bytes per f32

    // Initialize accumulator to zero
    %C_init = func.call @zero_C() : () -> !rt_C_f32

    // K-loop iteration 0: k = 0..16
    %A0 = func.call @load_A_f16(%A_ptr, %c0, %c0, %stride_AB) : (!sx2, index, index, index) -> !future_global_read
    %B0 = func.call @load_B_f16(%B_ptr, %c0, %c0, %stride_AB) : (!sx2, index, index, index) -> !future_global_read
    %C_after_k0 = func.call @mfma_f32_16x16x16_f16_future(%A0, %B0, %C_init) : (!future_global_read, !future_global_read, !rt_C_f32) -> !rt_C_f32

    // K-loop iteration 1: k = 16..32
    %A1 = func.call @load_A_f16(%A_ptr, %c0, %c16, %stride_AB) : (!sx2, index, index, index) -> !future_global_read
    %B1 = func.call @load_B_f16(%B_ptr, %c0, %c16, %stride_AB) : (!sx2, index, index, index) -> !future_global_read
    %C_final = func.call @mfma_f32_16x16x16_f16_future(%A1, %B1, %C_after_k0) : (!future_global_read, !future_global_read, !rt_C_f32) -> !rt_C_f32

    // Store result
    %store_tok = func.call @store_C_f32(%C_final, %C_ptr, %c0, %c0, %stride_C) : (!rt_C_f32, !sx2, index, index, index) -> !write_token
    amdgcn.wait deps %store_tok : !write_token

    amdgcn.end_kernel
  }
}
