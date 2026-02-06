// Kittens LDS transfer primitives for 16x16 tiles.
// Global Memory <-> LDS <-> Registers

// Register types
!sx2 = !amdgcn.sgpr<[? + 2]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>

// Kittens register tile types
!rt_A_f16 = !vx2
!rt_B_f16 = !vx2

// Descriptor types from indexing.mlir
!index_pair = !aster_utils.struct<i: index, j: index>
!index_descriptor_2level_2d = !aster_utils.struct<i: index, j: index, ii: index, jj: index, stride: index, elt_size_b: index>

amdgcn.library @kittens_lds_transfers isa = [#amdgcn.isa<cdna3>] {
  // From indexing.mlir
  func.func private @mfma_index_A_16x16xf16() -> !index_pair
  func.func private @tiled_matrix_offset(!index_descriptor_2level_2d) -> !v
  func.func private @alloc_vgprx2() -> !vx2

  // From lds_16x16.mlir
  func.func private @thread_lds_slice() -> (index, index)
  func.func private @lds_element_offset(index, index, index) -> index
  func.func private @lds_barrier()

  //===--------------------------------------------------------------------===//
  // Global -> LDS Transfers (Cooperative Loads)
  //===--------------------------------------------------------------------===//

  // Cooperative load: each of 64 threads loads 4 f16 (8 bytes) from global to LDS.
  func.func private @load_global_to_lds_f16(
      %lds_tile_base: index,
      %global_ptr: !sx2,
      %m: index,
      %n: index,
      %stride: index
  ) {
    %row, %col = func.call @thread_lds_slice() : () -> (index, index)

    // Compute global memory offset for this thread's slice
    %elt_size = arith.constant 2 : index  // f16 = 2 bytes
    %desc = aster_utils.struct_create(%m, %n, %row, %col, %stride, %elt_size)
        : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %global_off_vgpr = func.call @tiled_matrix_offset(%desc)
        : (!index_descriptor_2level_2d) -> !v

    // Load 8 bytes from global memory
    %c0_i32 = arith.constant 0 : i32
    %tmp_reg = func.call @alloc_vgprx2() : () -> !vx2
    %loaded, %tok_global = amdgcn.load global_load_dwordx2 dest %tmp_reg addr %global_ptr
        offset d(%global_off_vgpr) + c(%c0_i32)
        : dps(!vx2) ins(!sx2, !v, i32) -> !amdgcn.read_token<flat>
    amdgcn.wait deps %tok_global : !amdgcn.read_token<flat>

    // Compute LDS address and write
    %lds_offset_idx = func.call @lds_element_offset(%lds_tile_base, %row, %col)
        : (index, index, index) -> index
    %lds_offset_i32 = arith.index_cast %lds_offset_idx : index to i32
    %lds_addr = lsir.to_reg %lds_offset_i32 : i32 -> !v

    %c0_i32_2 = arith.constant 0 : i32
    %tok_lds = amdgcn.store ds_write_b64 data %loaded addr %lds_addr offset c(%c0_i32_2)
        : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>
    amdgcn.wait deps %tok_lds : !amdgcn.write_token<shared>

    return
  }

  //===--------------------------------------------------------------------===//
  // LDS -> Register Transfers
  //===--------------------------------------------------------------------===//

  // Load 16x16 f16 tile from LDS to register in MFMA A fragment layout.
  // Caller must ensure LDS data is ready (barrier before calling).
  func.func private @load_lds_to_register_A_f16(%lds_tile_base: index) -> !rt_A_f16 {
    %mfma_idx = func.call @mfma_index_A_16x16xf16() : () -> !index_pair
    %row, %col = aster_utils.struct_extract %mfma_idx ["i", "j"]
        : !index_pair -> index, index

    %lds_offset_idx = func.call @lds_element_offset(%lds_tile_base, %row, %col)
        : (index, index, index) -> index
    %lds_offset_i32 = arith.index_cast %lds_offset_idx : index to i32
    %lds_addr = lsir.to_reg %lds_offset_i32 : i32 -> !v

    %dst = func.call @alloc_vgprx2() : () -> !vx2
    %c0 = arith.constant 0 : i32
    %result, %tok = amdgcn.load ds_read_b64 dest %dst addr %lds_addr offset c(%c0)
        : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    amdgcn.wait deps %tok : !amdgcn.read_token<shared>

    return %result : !rt_A_f16
  }

  func.func private @load_lds_to_register_B_f16(%lds_tile_base: index) -> !rt_B_f16 {
    %result = func.call @load_lds_to_register_A_f16(%lds_tile_base)
        : (index) -> !rt_A_f16
    return %result : !rt_B_f16
  }

  //===--------------------------------------------------------------------===//
  // Register -> LDS Transfers
  //===--------------------------------------------------------------------===//

  func.func private @store_register_A_to_lds_f16(%tile: !rt_A_f16, %lds_tile_base: index) {
    %mfma_idx = func.call @mfma_index_A_16x16xf16() : () -> !index_pair
    %row, %col = aster_utils.struct_extract %mfma_idx ["i", "j"]
        : !index_pair -> index, index

    %lds_offset_idx = func.call @lds_element_offset(%lds_tile_base, %row, %col)
        : (index, index, index) -> index
    %lds_offset_i32 = arith.index_cast %lds_offset_idx : index to i32
    %lds_addr = lsir.to_reg %lds_offset_i32 : i32 -> !v

    %c0 = arith.constant 0 : i32
    %tok = amdgcn.store ds_write_b64 data %tile addr %lds_addr offset c(%c0)
        : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>
    amdgcn.wait deps %tok : !amdgcn.write_token<shared>

    return
  }

  func.func private @store_register_B_to_lds_f16(%tile: !rt_B_f16, %lds_tile_base: index) {
    func.call @store_register_A_to_lds_f16(%tile, %lds_tile_base)
        : (!rt_A_f16, index) -> ()
    return
  }

  //===--------------------------------------------------------------------===//
  // Convenience Wrappers
  //===--------------------------------------------------------------------===//

  // Global -> LDS -> Register in one call (single-buffer, no latency hiding)
  func.func private @load_global_to_register_A_via_lds_f16(
      %lds_tile_base: index,
      %global_ptr: !sx2,
      %m: index,
      %n: index,
      %stride: index
  ) -> !rt_A_f16 {
    func.call @load_global_to_lds_f16(%lds_tile_base, %global_ptr, %m, %n, %stride)
        : (index, !sx2, index, index, index) -> ()
    %result = func.call @load_lds_to_register_A_f16(%lds_tile_base)
        : (index) -> !rt_A_f16
    return %result : !rt_A_f16
  }

  func.func private @load_global_to_register_B_via_lds_f16(
      %lds_tile_base: index,
      %global_ptr: !sx2,
      %m: index,
      %n: index,
      %stride: index
  ) -> !rt_B_f16 {
    %result = func.call @load_global_to_register_A_via_lds_f16(
        %lds_tile_base, %global_ptr, %m, %n, %stride)
        : (index, !sx2, index, index, index) -> !rt_A_f16
    return %result : !rt_B_f16
  }
}
