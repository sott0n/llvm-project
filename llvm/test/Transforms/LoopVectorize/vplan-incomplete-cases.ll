; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --version 5
; RUN: opt -passes=loop-vectorize -S %s | FileCheck %s

; This test used to crash due to missing Or/Not cases in inferScalarTypeForRecipe.
define void @vplan_incomplete_cases_tc2(i8 %x, i8 %y) {
; CHECK-LABEL: define void @vplan_incomplete_cases_tc2(
; CHECK-SAME: i8 [[X:%.*]], i8 [[Y:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*]]:
; CHECK-NEXT:    br label %[[LOOP_HEADER:.*]]
; CHECK:       [[LOOP_HEADER]]:
; CHECK-NEXT:    [[IV:%.*]] = phi i8 [ [[IV_NEXT:%.*]], %[[LATCH:.*]] ], [ 0, %[[ENTRY]] ]
; CHECK-NEXT:    [[AND:%.*]] = and i8 [[X]], [[Y]]
; CHECK-NEXT:    [[EXTRACT_T:%.*]] = trunc i8 [[AND]] to i1
; CHECK-NEXT:    br i1 [[EXTRACT_T]], label %[[LATCH]], label %[[INDIRECT_LATCH:.*]]
; CHECK:       [[INDIRECT_LATCH]]:
; CHECK-NEXT:    br label %[[LATCH]]
; CHECK:       [[LATCH]]:
; CHECK-NEXT:    [[IV_NEXT]] = add i8 [[IV]], 1
; CHECK-NEXT:    [[ZEXT:%.*]] = zext i8 [[IV]] to i32
; CHECK-NEXT:    [[CMP:%.*]] = icmp ult i32 [[ZEXT]], 1
; CHECK-NEXT:    br i1 [[CMP]], label %[[LOOP_HEADER]], label %[[EXIT:.*]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    ret void
;
entry:
  br label %loop.header

loop.header:                                        ; preds = %latch, %entry
  %iv = phi i8 [ %iv.next, %latch ], [ 0, %entry ]
  %and = and i8 %x, %y
  %extract.t = trunc i8 %and to i1
  br i1 %extract.t, label %latch, label %indirect.latch

indirect.latch:                                     ; preds = %loop.header
  br label %latch

latch:                                              ; preds = %indirect.latch, loop.header
  %iv.next = add i8 %iv, 1
  %zext = zext i8 %iv to i32
  %cmp = icmp ult i32 %zext, 1
  br i1 %cmp, label %loop.header, label %exit

exit:                                               ; preds = %latch
  ret void
}

; This test used to crash due to missing the LogicalAnd case in inferScalarTypeForRecipe.
define void @vplan_incomplete_cases_tc3(i8 %x, i8 %y) {
; CHECK-LABEL: define void @vplan_incomplete_cases_tc3(
; CHECK-SAME: i8 [[X:%.*]], i8 [[Y:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*]]:
; CHECK-NEXT:    br label %[[LOOP_HEADER:.*]]
; CHECK:       [[LOOP_HEADER]]:
; CHECK-NEXT:    [[IV:%.*]] = phi i8 [ [[IV_NEXT:%.*]], %[[LATCH:.*]] ], [ 0, %[[ENTRY]] ]
; CHECK-NEXT:    [[AND:%.*]] = and i8 [[X]], [[Y]]
; CHECK-NEXT:    [[EXTRACT_T:%.*]] = trunc i8 [[AND]] to i1
; CHECK-NEXT:    br i1 [[EXTRACT_T]], label %[[LATCH]], label %[[INDIRECT_LATCH:.*]]
; CHECK:       [[INDIRECT_LATCH]]:
; CHECK-NEXT:    br label %[[LATCH]]
; CHECK:       [[LATCH]]:
; CHECK-NEXT:    [[IV_NEXT]] = add i8 [[IV]], 1
; CHECK-NEXT:    [[ZEXT:%.*]] = zext i8 [[IV]] to i32
; CHECK-NEXT:    [[CMP:%.*]] = icmp ult i32 [[ZEXT]], 2
; CHECK-NEXT:    br i1 [[CMP]], label %[[LOOP_HEADER]], label %[[EXIT:.*]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    ret void
;
entry:
  br label %loop.header

loop.header:                                        ; preds = %latch, %entry
  %iv = phi i8 [ %iv.next, %latch ], [ 0, %entry ]
  %and = and i8 %x, %y
  %extract.t = trunc i8 %and to i1
  br i1 %extract.t, label %latch, label %indirect.latch

indirect.latch:                                     ; preds = %loop.header
  br label %latch

latch:                                              ; preds = %indirect.latch, loop.header
  %iv.next = add i8 %iv, 1
  %zext = zext i8 %iv to i32
  %cmp = icmp ult i32 %zext, 2
  br i1 %cmp, label %loop.header, label %exit

exit:                                               ; preds = %latch
  ret void
}
