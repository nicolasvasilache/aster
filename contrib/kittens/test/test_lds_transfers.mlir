// Unit test for LDS transfer primitives
// RUN: aster-opt %s | FileCheck %s

// Type aliases
!sx2 = !amdgcn.sgpr<[? + 2]>
!vx2 = !amdgcn.vgpr<[? + 2]>
!rt_A_f16 = !vx2
!rt_B_f16 = !vx2

amdgcn.module @test_lds_transfers target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From kittens/lds_16x16.mlir
  func.func private @alloc_lds_1buffer() -> (index, index)

  // From kittens/lds_transfers.mlir
  func.func private @load_global_to_lds_f16(index, !sx2, index, index, index)
  func.func private @load_lds_to_register_A_f16(index) -> !rt_A_f16
  func.func private @load_lds_to_register_B_f16(index) -> !rt_B_f16
  func.func private @store_register_A_to_lds_f16(!rt_A_f16, index)
  func.func private @load_global_to_register_A_via_lds_f16(index, !sx2, index, index, index) -> !rt_A_f16

  // Test Global -> LDS transfer
  // CHECK-LABEL: @test_load_global_to_lds
  func.func @test_load_global_to_lds(%ptr: !sx2) {
    // Allocate LDS
    %A_base, %B_base = func.call @alloc_lds_1buffer() : () -> (index, index)

    // Tile position and stride
    %m = arith.constant 0 : index
    %n = arith.constant 0 : index
    %stride = arith.constant 64 : index  // 32 * 2 bytes for f16

    // CHECK: call @load_global_to_lds_f16
    func.call @load_global_to_lds_f16(%A_base, %ptr, %m, %n, %stride)
        : (index, !sx2, index, index, index) -> ()


    return
  }

  // Test LDS -> Register transfer (A tile)
  // CHECK-LABEL: @test_load_lds_to_register_A
  func.func @test_load_lds_to_register_A() -> !rt_A_f16 {
    // Allocate LDS
    %A_base, %B_base = func.call @alloc_lds_1buffer() : () -> (index, index)

    // CHECK: call @load_lds_to_register_A_f16
    %tile = func.call @load_lds_to_register_A_f16(%A_base)
        : (index) -> !rt_A_f16

    return %tile : !rt_A_f16
  }

  // Test LDS -> Register transfer (B tile)
  // CHECK-LABEL: @test_load_lds_to_register_B
  func.func @test_load_lds_to_register_B() -> !rt_B_f16 {
    // Allocate LDS
    %A_base, %B_base = func.call @alloc_lds_1buffer() : () -> (index, index)

    // CHECK: call @load_lds_to_register_B_f16
    %tile = func.call @load_lds_to_register_B_f16(%B_base)
        : (index) -> !rt_B_f16

    return %tile : !rt_B_f16
  }

  // Test Register -> LDS transfer
  // CHECK-LABEL: @test_store_register_to_lds
  func.func @test_store_register_to_lds(%tile: !rt_A_f16) {
    // Allocate LDS
    %A_base, %B_base = func.call @alloc_lds_1buffer() : () -> (index, index)

    // CHECK: call @store_register_A_to_lds_f16
    func.call @store_register_A_to_lds_f16(%tile, %A_base)
        : (!rt_A_f16, index) -> ()


    return
  }

  // Test convenience wrapper: Global -> LDS -> Register
  // CHECK-LABEL: @test_load_via_lds
  func.func @test_load_via_lds(%ptr: !sx2) -> !rt_A_f16 {
    // Allocate LDS
    %A_base, %B_base = func.call @alloc_lds_1buffer() : () -> (index, index)

    // Tile position and stride
    %m = arith.constant 0 : index
    %n = arith.constant 0 : index
    %stride = arith.constant 64 : index

    // CHECK: call @load_global_to_register_A_via_lds_f16
    %tile = func.call @load_global_to_register_A_via_lds_f16(
        %A_base, %ptr, %m, %n, %stride)
        : (index, !sx2, index, index, index) -> !rt_A_f16

    return %tile : !rt_A_f16
  }

  // Integration test: Full round-trip (Global -> LDS -> Register -> LDS)
  // CHECK-LABEL: @test_roundtrip
  func.func @test_roundtrip(%ptr: !sx2) -> !rt_A_f16 {
    // Allocate LDS
    %A_base, %B_base = func.call @alloc_lds_1buffer() : () -> (index, index)

    %m = arith.constant 0 : index
    %n = arith.constant 0 : index
    %stride = arith.constant 64 : index

    // Step 1: Global -> LDS
    // CHECK: call @load_global_to_lds_f16
    func.call @load_global_to_lds_f16(%A_base, %ptr, %m, %n, %stride)
        : (index, !sx2, index, index, index) -> ()

    // Step 2: LDS -> Register
    // CHECK: call @load_lds_to_register_A_f16
    %tile = func.call @load_lds_to_register_A_f16(%A_base)
        : (index) -> !rt_A_f16

    // Step 3: Register -> LDS (write to B buffer)
    // CHECK: call @store_register_A_to_lds_f16
    func.call @store_register_A_to_lds_f16(%tile, %B_base)
        : (!rt_A_f16, index) -> ()

    // Step 4: LDS -> Register (read back from B buffer)
    // CHECK: call @load_lds_to_register_A_f16
    %tile2 = func.call @load_lds_to_register_A_f16(%B_base)
        : (index) -> !rt_A_f16

    return %tile2 : !rt_A_f16
  }
}
