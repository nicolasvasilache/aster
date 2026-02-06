// Kittens GEMM kernel with triple-buffer LDS: C = A @ B^T
// A: 16xK (f16), B: 16xK (f16), C: 16x16 (f32)
//
// Tests three LDS addressing modes via {{LDS_MODE}}:
//   0 = non-padded (naive, HAS bank conflicts) - baseline
//   1 = padded (stride 17, avoids bank conflicts)
//   2 = XOR swizzle (avoids bank conflicts, no wasted space)
//
// Template parameters:
//   {{K}}         - K dimension (e.g., 48, 64, 96)
//   {{K_TILES}}   - Number of K tiles = K / 16
//   {{STRIDE_AB}} - Row stride in bytes for A and B = K * 2
//   {{LDS_MODE}}  - LDS addressing mode: 0=nopad, 1=padded, 2=xor_swizzle
//   {{LDS_BYTES}} - Total LDS bytes: 3072 (nopad/swizzle) or 3264 (padded)

// Type aliases
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>
!rt_A_f16 = !vx2
!rt_B_f16 = !vx2
!rt_C_f32 = !vx4

amdgcn.module @kittens_gemm_16x16xK_lds_3buf target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From kittens/tiles_16x16.mlir
  func.func private @zero_C() -> !rt_C_f32
  func.func private @mfma_f32_16x16x16_f16(!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
  func.func private @store_C_f32(!rt_C_f32, !sx2, index, index, index)

  // From kittens/lds_16x16.mlir - allocation (mode-specific)
  func.func private @alloc_lds_3buffer_nopad() -> (index, index, index, index, index, index)
  func.func private @alloc_lds_3buffer() -> (index, index, index, index, index, index)
  func.func private @alloc_lds_3buffer_xor_swizzle() -> (index, index, index, index, index, index)
  func.func private @lds_barrier()

  // From kittens/lds_transfers.mlir - transfers (mode-specific)
  // Mode 0: non-padded
  func.func private @load_global_to_lds_nopad_f16(index, !sx2, index, index, index)
  func.func private @load_lds_to_register_A_nopad_f16(index) -> !rt_A_f16
  func.func private @load_lds_to_register_B_nopad_f16(index) -> !rt_B_f16
  // Mode 1: padded
  func.func private @load_global_to_lds_f16(index, !sx2, index, index, index)
  func.func private @load_lds_to_register_A_f16(index) -> !rt_A_f16
  func.func private @load_lds_to_register_B_f16(index) -> !rt_B_f16
  // Mode 2: XOR swizzle
  func.func private @load_global_to_lds_xor_swizzle_f16(index, !sx2, index, index, index)
  func.func private @load_lds_to_register_A_xor_swizzle_f16(index) -> !rt_A_f16
  func.func private @load_lds_to_register_B_xor_swizzle_f16(index) -> !rt_B_f16

  // Helper: allocate triple-buffer LDS based on mode
  func.func private @alloc_lds(%mode: index)
      -> (index, index, index, index, index, index) {
    %a0, %b0, %a1, %b1, %a2, %b2 = scf.index_switch %mode
        -> index, index, index, index, index, index
    case 0 {
      %r0, %r1, %r2, %r3, %r4, %r5 = func.call @alloc_lds_3buffer_nopad()
          : () -> (index, index, index, index, index, index)
      scf.yield %r0, %r1, %r2, %r3, %r4, %r5 : index, index, index, index, index, index
    }
    case 1 {
      %r0, %r1, %r2, %r3, %r4, %r5 = func.call @alloc_lds_3buffer()
          : () -> (index, index, index, index, index, index)
      scf.yield %r0, %r1, %r2, %r3, %r4, %r5 : index, index, index, index, index, index
    }
    default {
      %r0, %r1, %r2, %r3, %r4, %r5 = func.call @alloc_lds_3buffer_xor_swizzle()
          : () -> (index, index, index, index, index, index)
      scf.yield %r0, %r1, %r2, %r3, %r4, %r5 : index, index, index, index, index, index
    }
    return %a0, %b0, %a1, %b1, %a2, %b2 : index, index, index, index, index, index
  }

  // Helper: cooperative global -> LDS load based on mode
  func.func private @global_to_lds(
      %mode: index, %lds_base: index, %ptr: !sx2,
      %m: index, %n: index, %stride: index
  ) {
    scf.index_switch %mode
    case 0 {
      func.call @load_global_to_lds_nopad_f16(%lds_base, %ptr, %m, %n, %stride)
          : (index, !sx2, index, index, index) -> ()
      scf.yield
    }
    case 1 {
      func.call @load_global_to_lds_f16(%lds_base, %ptr, %m, %n, %stride)
          : (index, !sx2, index, index, index) -> ()
      scf.yield
    }
    default {
      func.call @load_global_to_lds_xor_swizzle_f16(%lds_base, %ptr, %m, %n, %stride)
          : (index, !sx2, index, index, index) -> ()
      scf.yield
    }
    return
  }

  // Helper: LDS -> register tile load based on mode
  func.func private @lds_to_reg_A(%mode: index, %lds_base: index) -> !rt_A_f16 {
    %tile = scf.index_switch %mode -> !rt_A_f16
    case 0 {
      %t = func.call @load_lds_to_register_A_nopad_f16(%lds_base) : (index) -> !rt_A_f16
      scf.yield %t : !rt_A_f16
    }
    case 1 {
      %t = func.call @load_lds_to_register_A_f16(%lds_base) : (index) -> !rt_A_f16
      scf.yield %t : !rt_A_f16
    }
    default {
      %t = func.call @load_lds_to_register_A_xor_swizzle_f16(%lds_base) : (index) -> !rt_A_f16
      scf.yield %t : !rt_A_f16
    }
    return %tile : !rt_A_f16
  }

  func.func private @lds_to_reg_B(%mode: index, %lds_base: index) -> !rt_B_f16 {
    %tile = scf.index_switch %mode -> !rt_B_f16
    case 0 {
      %t = func.call @load_lds_to_register_B_nopad_f16(%lds_base) : (index) -> !rt_B_f16
      scf.yield %t : !rt_B_f16
    }
    case 1 {
      %t = func.call @load_lds_to_register_B_f16(%lds_base) : (index) -> !rt_B_f16
      scf.yield %t : !rt_B_f16
    }
    default {
      %t = func.call @load_lds_to_register_B_xor_swizzle_f16(%lds_base) : (index) -> !rt_B_f16
      scf.yield %t : !rt_B_f16
    }
    return %tile : !rt_B_f16
  }

  // GEMM kernel: C[16x16] = A[16xK] @ B[16xK]^T using triple-buffer LDS
  amdgcn.kernel @gemm_16x16xK_lds_3buf arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {shared_memory_size = {{LDS_BYTES}} : i32} {
    %A_ptr = amdgcn.load_arg 0 : !sx2
    %B_ptr = amdgcn.load_arg 1 : !sx2
    %C_ptr = amdgcn.load_arg 2 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    // Constants
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c16 = arith.constant 16 : index
    %mode = arith.constant {{LDS_MODE}} : index

    // Strides in bytes
    %stride_AB = arith.constant {{STRIDE_AB}} : index  // K * 2 bytes per f16
    %stride_C = arith.constant 64 : index              // 16 * 4 bytes per f32

    // Number of K tiles (K / 16)
    %K_tiles = arith.constant {{K_TILES}} : index

    // Allocate LDS: triple buffer for A and B tiles
    %lds_A0, %lds_B0, %lds_A1, %lds_B1, %lds_A2, %lds_B2 = func.call @alloc_lds(%mode)
        : (index) -> (index, index, index, index, index, index)

    // Initialize accumulator to zero
    %C_init = func.call @zero_C() : () -> !rt_C_f32

    // === Prologue: prefetch iterations 0 and 1 ===
    func.call @global_to_lds(%mode, %lds_A0, %A_ptr, %c0, %c0, %stride_AB)
        : (index, index, !sx2, index, index, index) -> ()
    func.call @global_to_lds(%mode, %lds_B0, %B_ptr, %c0, %c0, %stride_AB)
        : (index, index, !sx2, index, index, index) -> ()

    // Prefetch iteration 1 into buffer 1 (if K_tiles > 1)
    %has_iter1 = arith.cmpi ugt, %K_tiles, %c1 : index
    scf.if %has_iter1 {
      func.call @global_to_lds(%mode, %lds_A1, %A_ptr, %c0, %c16, %stride_AB)
          : (index, index, !sx2, index, index, index) -> ()
      func.call @global_to_lds(%mode, %lds_B1, %B_ptr, %c0, %c16, %stride_AB)
          : (index, index, !sx2, index, index, index) -> ()
    }

    // K-loop: triple-buffered iteration over K dimension
    // buf_idx cycles 0->1->2->0->... as an iter_arg
    %C_final, %buf_final = scf.for %k = %c0 to %K_tiles step %c1
        iter_args(%acc = %C_init, %buf_idx = %c0) -> (!rt_C_f32, index) {
      // === Buffer Selection (3-way mux on buf_idx) ===
      // Each cmpi must be consumed by its select before the next cmpi,
      // because i1 values live in SCC and cannot overlap.
      %is_buf1 = arith.cmpi eq, %buf_idx, %c1 : index
      %lds_A_12 = arith.select %is_buf1, %lds_A1, %lds_A2 : index
      %lds_B_12 = arith.select %is_buf1, %lds_B1, %lds_B2 : index
      %is_buf0 = arith.cmpi eq, %buf_idx, %c0 : index
      %lds_A_cur = arith.select %is_buf0, %lds_A0, %lds_A_12 : index
      %lds_B_cur = arith.select %is_buf0, %lds_B0, %lds_B_12 : index

      // === Prefetch k+2 (if within bounds) ===
      %k_plus2 = arith.addi %k, %c2 : index
      %has_prefetch = arith.cmpi ult, %k_plus2, %K_tiles : index

      scf.if %has_prefetch {
        // Prefetch target buffer: (buf_idx + 2) % 3
        %pf_raw = arith.addi %buf_idx, %c2 : index
        %pf_ge3 = arith.cmpi uge, %pf_raw, %c3 : index
        %pf_wrapped = arith.subi %pf_raw, %c3 : index
        %pf_buf_idx = arith.select %pf_ge3, %pf_wrapped, %pf_raw : index
        %pf_is_buf1 = arith.cmpi eq, %pf_buf_idx, %c1 : index
        %pf_A_12 = arith.select %pf_is_buf1, %lds_A1, %lds_A2 : index
        %pf_B_12 = arith.select %pf_is_buf1, %lds_B1, %lds_B2 : index
        %pf_is_buf0 = arith.cmpi eq, %pf_buf_idx, %c0 : index
        %pf_A = arith.select %pf_is_buf0, %lds_A0, %pf_A_12 : index
        %pf_B = arith.select %pf_is_buf0, %lds_B0, %pf_B_12 : index

        // Compute column offset for iteration k+2
        %pf_offset = arith.muli %k_plus2, %c16 : index

        // Async load k+2 into prefetch buffer
        func.call @global_to_lds(%mode, %pf_A, %A_ptr, %c0, %pf_offset, %stride_AB)
            : (index, index, !sx2, index, index, index) -> ()
        func.call @global_to_lds(%mode, %pf_B, %B_ptr, %c0, %pf_offset, %stride_AB)
            : (index, index, !sx2, index, index, index) -> ()
      }

      // === Wait for current iteration k's data ===
      func.call @lds_barrier() : () -> ()

      // === Load LDS -> Register ===
      %A_tile = func.call @lds_to_reg_A(%mode, %lds_A_cur) : (index, index) -> !rt_A_f16
      %B_tile = func.call @lds_to_reg_B(%mode, %lds_B_cur) : (index, index) -> !rt_B_f16

      // === Compute ===
      %new_acc = func.call @mfma_f32_16x16x16_f16(%A_tile, %B_tile, %acc)
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32

      // Advance buffer index: (buf_idx + 1) % 3 via wrap
      %next_raw = arith.addi %buf_idx, %c1 : index
      %next_is_3 = arith.cmpi eq, %next_raw, %c3 : index
      %next_buf = arith.select %next_is_3, %c0, %next_raw : index

      scf.yield %new_acc, %next_buf : !rt_C_f32, index
    }

    // Store result to global memory
    func.call @store_C_f32(%C_final, %C_ptr, %c0, %c0, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> ()

    amdgcn.end_kernel
  }
}
