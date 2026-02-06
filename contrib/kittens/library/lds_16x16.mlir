// Kittens LDS (Local Data Share / Shared Memory) allocation and addressing
// primitives for 16x16 tiles.
//
// Design: 16x16 f16 tiles with padded stride (17 columns) for bank conflict
// avoidance. Supports multi-buffering (1, 2, or 3 buffers) for latency hiding.

amdgcn.library @kittens_lds_16x16 isa = [#amdgcn.isa<cdna3>] {
  // From indexing.mlir
  func.func private @lane_id() -> index

  //===--------------------------------------------------------------------===//
  // LDS Allocation Functions
  //===--------------------------------------------------------------------===//

  // Allocate LDS for 1-buffer configuration (baseline, no latency hiding)
  // Returns: (A_buffer_0, B_buffer_0) as LDS allocation handles
  //
  // Layout:
  //   A[0]: size 544 bytes
  //   B[0]: size 544 bytes
  //   Total: 1,088 bytes
  func.func private @alloc_lds_1buffer() -> (index, index) {
    %A_base = amdgcn.alloc_lds 544
    %B_base = amdgcn.alloc_lds 544

    %A_off = amdgcn.get_lds_offset %A_base : index
    %B_off = amdgcn.get_lds_offset %B_base : index

    return %A_off, %B_off : index, index
  }

  // Allocate LDS for 2-buffer configuration (double buffering / ping-pong)
  // Returns: (A_buffer_0, B_buffer_0, A_buffer_1, B_buffer_1) as LDS offsets
  //
  // Layout:
  //   A[0]: size 544 bytes  (ping)
  //   B[0]: size 544 bytes  (ping)
  //   A[1]: size 544 bytes  (pong)
  //   B[1]: size 544 bytes  (pong)
  //   Total: 2,176 bytes
  func.func private @alloc_lds_2buffer() -> (index, index, index, index) {
    %A0_alloc = amdgcn.alloc_lds 544
    %B0_alloc = amdgcn.alloc_lds 544
    %A1_alloc = amdgcn.alloc_lds 544
    %B1_alloc = amdgcn.alloc_lds 544

    %A0 = amdgcn.get_lds_offset %A0_alloc : index
    %B0 = amdgcn.get_lds_offset %B0_alloc : index
    %A1 = amdgcn.get_lds_offset %A1_alloc : index
    %B1 = amdgcn.get_lds_offset %B1_alloc : index

    return %A0, %B0, %A1, %B1 : index, index, index, index
  }

  // Allocate LDS for 3-buffer configuration (triple buffering)
  // Returns: (A[0], B[0], A[1], B[1], A[2], B[2]) as LDS offsets
  //
  // Layout:
  //   A[0]: size 544 bytes
  //   B[0]: size 544 bytes
  //   A[1]: size 544 bytes
  //   B[1]: size 544 bytes
  //   A[2]: size 544 bytes
  //   B[2]: size 544 bytes
  //   Total: 3,264 bytes
  func.func private @alloc_lds_3buffer() -> (index, index, index, index, index, index) {
    %A0_alloc = amdgcn.alloc_lds 544
    %B0_alloc = amdgcn.alloc_lds 544
    %A1_alloc = amdgcn.alloc_lds 544
    %B1_alloc = amdgcn.alloc_lds 544
    %A2_alloc = amdgcn.alloc_lds 544
    %B2_alloc = amdgcn.alloc_lds 544

    %A0 = amdgcn.get_lds_offset %A0_alloc : index
    %B0 = amdgcn.get_lds_offset %B0_alloc : index
    %A1 = amdgcn.get_lds_offset %A1_alloc : index
    %B1 = amdgcn.get_lds_offset %B1_alloc : index
    %A2 = amdgcn.get_lds_offset %A2_alloc : index
    %B2 = amdgcn.get_lds_offset %B2_alloc : index

    return %A0, %B0, %A1, %B1, %A2, %B2 : index, index, index, index, index, index
  }

  //===--------------------------------------------------------------------===//
  // LDS Addressing Functions
  //===--------------------------------------------------------------------===//

  // Compute LDS byte offset for element at (row, col) within a tile.
  // Layout: 16 rows x 17 columns (padded) x 2 bytes (f16)
  // offset = tile_base + row * 34 + col * 2
  func.func private @lds_element_offset(
      %tile_base: index,
      %row: index,
      %col: index
  ) -> index {
    %offset = affine.apply affine_map<()[base, row, col] -> (base + row * 34 + col * 2)>
        ()[%tile_base, %row, %col]
    return %offset : index
  }


  //===--------------------------------------------------------------------===//
  // Thread-to-Element Mapping
  //===--------------------------------------------------------------------===//

  // Map lane ID to (row, col) for cooperative LDS loads of a 16x16 tile.
  // 64 lanes, 4 elements/lane, row-major: row = lane floordiv 4, col = (lane mod 4) * 4
  func.func private @thread_lds_slice() -> (index, index) {
    %lane = func.call @lane_id() : () -> index
    %row = affine.apply affine_map<()[lid] -> (lid floordiv 4)>()[%lane]
    %col = affine.apply affine_map<()[lid] -> ((lid mod 4) * 4)>()[%lane]
    return %row, %col : index, index
  }}
