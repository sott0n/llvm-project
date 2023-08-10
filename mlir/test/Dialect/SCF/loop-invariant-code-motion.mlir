// RUN: mlir-opt %s -scf-loop-invariant-code-motion -split-input-file | FileCheck %s

// CHECK-LABEL: func @loop_having_invariant_code
func.func @loop_having_invariant_code() {
  %m = memref.alloc() : memref<10xf32>
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c10 = arith.constant 10 : index

  scf.for %arg0 = %c0 to %c10 step %c2 {
    %cf7 = arith.constant 7.0 : f32
    %cf8 = arith.constant 8.0 : f32
    %v0 = arith.addf %cf7, %cf8 : f32
    memref.store %v0, %m[%arg0] : memref<10xf32>
  }
  return

  //      CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %[[cst_0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[cst_2:.*]] = arith.constant 2 : index
  // CHECK-NEXT: %[[cst_10:.*]] = arith.constant 10 : index
  // CHECK-NEXT: %[[cst_7:.*]] = arith.constant 7.000000e+00 : f32
  // CHECK-NEXT: %[[cst_8:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: arith.addf %[[cst_7]], %[[cst_8]] : f32
  // CHECK-NEXT: scf.for %{{.*}} = %[[cst_0]] to %[[cst_10]] step %[[cst_2]] {
  // CHECK-NEXT:   memref.store
  // CHECK-NEXT: }
}

// -----

// CHECK-LABEL: func @nested_loop_both_having_invariant_code
func.func @nested_loop_both_having_invariant_code() {
  %m = memref.alloc() : memref<10xf32>
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c10 = arith.constant 10 : index

  scf.for %arg0 = %c0 to %c10 step %c2 {
    %cf7 = arith.constant 7.0 : f32
    %cf8 = arith.constant 8.0 : f32
    %v0 = arith.addf %cf7, %cf8 : f32
    scf.for %arg1 = %c0 to %c10 step %c2 {
      memref.store %v0, %m[%arg0] : memref<10xf32>
    }
  }
  return

  //      CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %[[cst_0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[cst_2:.*]] = arith.constant 2 : index
  // CHECK-NEXT: %[[cst_10:.*]] = arith.constant 10 : index
  // CHECK-NEXT: %[[cst_7:.*]] = arith.constant 7.000000e+00 : f32
  // CHECK-NEXT: %[[cst_8:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: arith.addf %[[cst_7]], %[[cst_8]] : f32
  // CHECK-NEXT: scf.for %{{.*}} = %[[cst_0]] to %[[cst_10]] step %[[cst_2]] {
  // CHECK-NEXT:   scf.for %{{.*}} = %[[cst_0]] to %[[cst_10]] step %[[cst_2]] {
  // CHECK-NEXT:     memref.store
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
}

// -----

// CHECK-LABEL: @single_loop_nothing_invariant
func.func @single_loop_nothing_invariant() {
  %m1 = memref.alloc() : memref<10xf32>
  %m2 = memref.alloc() : memref<11xf32>
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c10 = arith.constant 10 : index

  scf.for %i = %c0 to %c10 step %c2 {
    %v0 = memref.load %m1[%i] : memref<10xf32>
    %v1 = memref.load %m2[%i] : memref<11xf32>
    %v2 = arith.addf %v0, %v1 : f32
    memref.store %v2, %m1[%i] : memref<10xf32>
  }
  return

  //      CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: memref.alloc() : memref<11xf32>
  // CHECK-NEXT: %[[cst_0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[cst_2:.*]] = arith.constant 2 : index
  // CHECK-NEXT: %[[cst_10:.*]] = arith.constant 10 : index
  // CHECK-NEXT: scf.for %[[i:.*]] = %[[cst_0]] to %[[cst_10]] step %[[cst_2]] {
  // CHECK-NEXT:   memref.load %{{.*}} : memref<10xf32>
  // CHECK-NEXT:   memref.load %{{.*}} : memref<11xf32>
  // CHECK-NEXT:   arith.addf
  // CHECK-NEXT:   memref.store %{{.*}}, %{{.*}}[%[[i]]] : memref<10xf32>
  // CHECK-NEXT: }
}

// -----

// CHECK-LABEL: @invariant_if
func.func @invariant_if(%arg0: i1, %arg1: f32) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c10 = arith.constant 10 : index
  %cf8 = arith.constant 8.0 : f32

  scf.for %i = %c0 to %c10 step %c2 {
    scf.for %j = %c0 to %c10 step %c2 {
      scf.if %arg0 {
        %0 = arith.addf %arg1, %arg1 : f32
      }
    }
  }
  return

  //      CHECK: %[[cst_0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[cst_2:.*]] = arith.constant 2 : index
  // CHECK-NEXT: %[[cst_10:.*]] = arith.constant 10 : index
  // CHECK-NEXT: %[[cst_8:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: scf.if %{{.*}} {
  // CHECK-NEXT:   %{{.*}} = arith.addf
  // CHECK-NEXT: }
  // CHECK-NEXT: scf.for %[[i:.*]] = %[[cst_0]] to %[[cst_10]] step %[[cst_2]] {
  // CHECK-NEXT: }
  // CHECK-NEXT: scf.for %[[i:.*]] = %[[cst_0]] to %[[cst_10]] step %[[cst_2]] {
  // CHECK-NEXT: }
}

// -----

// CHECK-LABEL: @invariant_if_else
func.func @invariant_if_else(%arg0: i1, %arg1: f32) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c10 = arith.constant 10 : index
  %cf8 = arith.constant 8.0 : f32

  scf.for %i = %c0 to %c10 step %c2 {
    scf.for %j = %c0 to %c10 step %c2 {
      scf.if %arg0 {
        %0 = arith.addf %arg1, %arg1 : f32
      } else {
        %1 = arith.addf %arg1, %arg1 : f32
      }
    }
  }
  return

  //      CHECK: %[[cst_0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[cst_2:.*]] = arith.constant 2 : index
  // CHECK-NEXT: %[[cst_10:.*]] = arith.constant 10 : index
  // CHECK-NEXT: %[[cst_8:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: scf.if %{{.*}} {
  // CHECK-NEXT:   %{{.*}} = arith.addf
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   %{{.*}} = arith.addf
  // CHECK-NEXT: }
  // CHECK-NEXT: scf.for %[[i:.*]] = %[[cst_0]] to %[[cst_10]] step %[[cst_2]] {
  // CHECK-NEXT: }
  // CHECK-NEXT: scf.for %[[i:.*]] = %[[cst_0]] to %[[cst_10]] step %[[cst_2]] {
  // CHECK-NEXT: }
}

// -----

// CHECK-LABEL: @not_invariant_if
func.func @not_invariant_if(%arg0: i1) {
  %c0 = arith.constant 0 : i64
  %c2 = arith.constant 2 : i64
  %c10 = arith.constant 10 : i64

  scf.for %i = %c0 to %c10 step %c2 : i64 {
    scf.for %j = %c0 to %c10 step %c2 : i64 {
      scf.if %arg0 {
        %0 = arith.addi %i, %j : i64
      }
    }
  }
  return

  //      CHECK: %[[cst_0:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[cst_2:.*]] = arith.constant 2 : i64
  // CHECK-NEXT: %[[cst_10:.*]] = arith.constant 10 : i64
  // CHECK-NEXT: scf.for %[[i:.*]] = %[[cst_0]] to %[[cst_10]] step %[[cst_2]] : i64 {
  // CHECK-NEXT:   scf.for %[[i:.*]] = %[[cst_0]] to %[[cst_10]] step %[[cst_2]] : i64 {
  // CHECK-NEXT:     scf.if %{{.*}} {
  // CHECK-NEXT:       %{{.*}} = arith.addi
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
}

// -----

// CHECK-LABEL: @not_invariant_if_else
func.func @not_invariant_if_else(%arg0: i1) {
  %c0 = arith.constant 0 : i64
  %c2 = arith.constant 2 : i64
  %c10 = arith.constant 10 : i64

  scf.for %i = %c0 to %c10 step %c2 : i64 {
    scf.for %j = %c0 to %c10 step %c2 : i64 {
      scf.if %arg0 {
        %0 = arith.addi %i, %j : i64
      } else {
        %0 = arith.addi %i, %j : i64
      }
    }
  }
  return

  //      CHECK: %[[cst_0:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[cst_2:.*]] = arith.constant 2 : i64
  // CHECK-NEXT: %[[cst_10:.*]] = arith.constant 10 : i64
  // CHECK-NEXT: scf.for %[[i:.*]] = %[[cst_0]] to %[[cst_10]] step %[[cst_2]] : i64 {
  // CHECK-NEXT:   scf.for %[[i:.*]] = %[[cst_0]] to %[[cst_10]] step %[[cst_2]] : i64 {
  // CHECK-NEXT:     scf.if %{{.*}} {
  // CHECK-NEXT:       %{{.*}} = arith.addi
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       %{{.*}} = arith.addi
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
}

// -----

// CHECK-LABEL: func @parallel_loop
func.func @parallel_loop(%arg0 : index, %arg1 : index, %arg2 : index,
                        %arg3 : index, %arg4 : index) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c10 = arith.constant 10 : index

  scf.for %i = %c0 to %c10 step %c2 {
    %step = arith.constant 1 : index
    scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                                            step (%arg4, %step) {
      %min_cmp = arith.cmpi slt, %i0, %i1 : index
      %min = arith.select %min_cmp, %i0, %i1 : index
      %max_cmp = arith.cmpi sge, %i2, %i1 : index
      %max = arith.select %max_cmp, %i0, %i1 : index
      %zero = arith.constant 0.0 : f32
      %int_zero = arith.constant 0 : i32
      %red:2 = scf.parallel (%i2) = (%min) to (%max) step (%i1)
                                        init (%zero, %int_zero) -> (f32, i32) {
        %one = arith.constant 1.0 : f32
        scf.reduce(%one) : f32 {
          ^bb0(%lhs : f32, %rhs: f32):
            %res = arith.addf %lhs, %rhs : f32
            scf.reduce.return %res : f32
        }
        %int_one = arith.constant 1 : i32
        scf.reduce(%int_one) : i32 {
          ^bb0(%lhs : i32, %rhs: i32):
            %res = arith.muli %lhs, %rhs : i32
            scf.reduce.return %res : i32
        }
      }
    }
  }
  return

  //  CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]:
  //  CHECK-SAME: %[[ARG1:[A-Za-z0-9]+]]:
  //  CHECK-SAME: %[[ARG2:[A-Za-z0-9]+]]:
  //  CHECK-SAME: %[[ARG3:[A-Za-z0-9]+]]:
  //  CHECK-SAME: %[[ARG4:[A-Za-z0-9]+]]:
  //      CHECK: %[[cst_0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[cst_2:.*]] = arith.constant 2 : index
  // CHECK-NEXT: %[[cst_10:.*]] = arith.constant 10 : index
  // CHECK-NEXT: %[[STEP:.*]] = arith.constant 1 : index
  // CHECK-NEXT:   scf.for %{{.*}} = %[[cst_0]] to %[[cst_10]] step %[[cst_2]] {
  // CHECK-NEXT:     scf.parallel (%[[I0:.*]], %[[I1:.*]]) = (%[[ARG0]], %[[ARG1]]) to
  //      CHECK:       (%[[ARG2]], %[[ARG3]]) step (%[[ARG4]], %[[STEP]]) {
  // CHECK-NEXT:     %[[MIN_CMP:.*]] = arith.cmpi slt, %[[I0]], %[[I1]] : index
  // CHECK-NEXT:     %[[MIN:.*]] = arith.select %[[MIN_CMP]], %[[I0]], %[[I1]] : index
  // CHECK-NEXT:     %[[MAX_CMP:.*]] = arith.cmpi sge, %[[I0]], %[[I1]] : index
  // CHECK-NEXT:     %[[MAX:.*]] = arith.select %[[MAX_CMP]], %[[I0]], %[[I1]] : index
  // CHECK-NEXT:     %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK-NEXT:     %[[INT_ZERO:.*]] = arith.constant 0 : i32
  // CHECK-NEXT:     scf.parallel (%{{.*}}) = (%[[MIN]]) to (%[[MAX]])
  // CHECK-SAME:          step (%[[I1]])
  // CHECK-SAME:          init (%[[ZERO]], %[[INT_ZERO]]) -> (f32, i32) {
  // CHECK-NEXT:       %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK-NEXT:       scf.reduce(%[[ONE]]) : f32 {
  // CHECK-NEXT:       ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32):
  // CHECK-NEXT:         %[[RES:.*]] = arith.addf %[[LHS]], %[[RHS]] : f32
  // CHECK-NEXT:         scf.reduce.return %[[RES]] : f32
  // CHECK-NEXT:       }
  // CHECK-NEXT:       %[[INT_ONE:.*]] = arith.constant 1 : i32
  // CHECK-NEXT:       scf.reduce(%[[INT_ONE]]) : i32 {
  // CHECK-NEXT:       ^bb0(%[[LHS:.*]]: i32, %[[RHS:.*]]: i32):
  // CHECK-NEXT:         %[[RES:.*]] = arith.muli %[[LHS]], %[[RHS]] : i32
  // CHECK-NEXT:         scf.reduce.return %[[RES]] : i32
  // CHECK-NEXT:       }
  // CHECK-NEXT:       scf.yield
  // CHECK-NEXT:     }
  // CHECK-NEXT:     scf.yield
}
