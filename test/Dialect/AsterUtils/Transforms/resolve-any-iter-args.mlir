// RUN: aster-opt %s --aster-resolve-any-iter-args | FileCheck %s

// CHECK-LABEL: func.func @basic_any_iter_arg
// CHECK-SAME:    %[[INIT:.*]]: i32
// CHECK:         %[[FOR:.*]] = scf.for {{.*}} iter_args(%[[ARG:.*]] = %[[INIT]]) -> (i32)
// CHECK:           %[[NEXT:.*]] = arith.addi %[[ARG]], %[[ARG]] : i32
// CHECK:           scf.yield %[[NEXT]] : i32
// CHECK:         return %[[FOR]] : i32
func.func @basic_any_iter_arg(%init: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %init_any = aster_utils.to_any %init : i32
  %result_any = scf.for %i = %c0 to %c10 step %c1 iter_args(%arg = %init_any) -> (!aster_utils.any) {
    %val = aster_utils.from_any %arg : i32
    %next = arith.addi %val, %val : i32
    %next_any = aster_utils.to_any %next : i32
    scf.yield %next_any : !aster_utils.any
  }
  %result = aster_utils.from_any %result_any : i32
  return %result : i32
}

// Mixed: one any-typed iter_arg, one concrete. Only the any one is specialized.
// CHECK-LABEL: func.func @mixed_any_and_concrete
// CHECK-SAME:    %[[S:.*]]: i32, %[[V:.*]]: f32
// CHECK:         scf.for {{.*}} iter_args(%[[A:.*]] = %[[S]], %[[B:.*]] = %[[V]]) -> (i32, f32)
// CHECK:           scf.yield %[[A]], %[[B]] : i32, f32
func.func @mixed_any_and_concrete(%s: i32, %v: f32) -> (i32, f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %s_any = aster_utils.to_any %s : i32
  %result:2 = scf.for %i = %c0 to %c10 step %c1 iter_args(%a = %s_any, %b = %v) -> (!aster_utils.any, f32) {
    %a_val = aster_utils.from_any %a : i32
    %a_any = aster_utils.to_any %a_val : i32
    scf.yield %a_any, %b : !aster_utils.any, f32
  }
  %r0 = aster_utils.from_any %result#0 : i32
  return %r0, %result#1 : i32, f32
}

// No any-typed iter_args: pass should not modify.
// CHECK-LABEL: func.func @no_any_unchanged
// CHECK:         scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (i32)
// CHECK-NOT:     aster_utils.any
func.func @no_any_unchanged(%init: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%v = %init) -> (i32) {
    scf.yield %v : i32
  }
  return %result : i32
}

// Rotation pattern from SCF pipeliner: some iter_args are consumed (from_any),
// some are passthrough (yielded as block args to rotate the prefetch buffer).
// CHECK-LABEL: func.func @rotation_pattern
// CHECK-SAME:    %[[A:.*]]: i32, %[[B:.*]]: i32, %[[C:.*]]: i32
// CHECK:         scf.for {{.*}} iter_args(%[[X:.*]] = %[[C]], %[[Y:.*]] = %[[B]], %[[Z:.*]] = %[[A]])
// CHECK-SAME:      -> (i32, i32, i32)
// CHECK:           %[[NEW:.*]] = arith.addi %[[Z]], %[[Z]] : i32
// CHECK:           scf.yield %[[NEW]], %[[X]], %[[Y]] : i32, i32, i32
func.func @rotation_pattern(%a: i32, %b: i32, %c: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %a_any = aster_utils.to_any %a : i32
  %b_any = aster_utils.to_any %b : i32
  %c_any = aster_utils.to_any %c : i32
  // 3-deep rotation buffer: consume oldest (arg2), shift others, produce new
  %result:3 = scf.for %i = %c0 to %c10 step %c1
      iter_args(%x = %c_any, %y = %b_any, %z = %a_any) -> (!aster_utils.any, !aster_utils.any, !aster_utils.any) {
    %consumed = aster_utils.from_any %z : i32
    %new_val = arith.addi %consumed, %consumed : i32
    %new_any = aster_utils.to_any %new_val : i32
    // Rotate: new -> x, old x -> y, old y -> z (consumed and dropped)
    scf.yield %new_any, %x, %y : !aster_utils.any, !aster_utils.any, !aster_utils.any
  }
  %final = aster_utils.from_any %result#2 : i32
  return %final : i32
}

// Init is not to_any: should not specialize this iter_arg.
// CHECK-LABEL: func.func @unresolvable_any
// CHECK:         scf.for {{.*}} iter_args({{.*}}) -> (!aster_utils.any)
func.func @unresolvable_any(%init_any: !aster_utils.any) -> !aster_utils.any {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%arg = %init_any) -> (!aster_utils.any) {
    scf.yield %arg : !aster_utils.any
  }
  return %result : !aster_utils.any
}
