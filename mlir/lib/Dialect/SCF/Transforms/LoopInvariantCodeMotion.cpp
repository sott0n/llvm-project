//===- LoopInvariantCodeMotion.cpp - Code to perform loop fusion-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop invariant code motion.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_SCFLOOPINVARIANTCODEMOTION
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "scf-licm"

using namespace mlir;
using namespace mlir::scf;

namespace {

static bool
checkInvarianceOfNestedIfOp(IfOp ifOp, Value indVar, ValueRange iterArgs,
                           SmallPtrSetImpl<Operation *> &opsWithUsers,
                           SmallPtrSetImpl<Operation *> &opsToHoist);
static bool
areAllOpsInTheBlockListInvariant(Region &blockList, Value indVar,
                                 ValueRange iterArgs,
                                 SmallPtrSetImpl<Operation *> &opsWithUsers,
                                 SmallPtrSetImpl<Operation *> &opsToHoist);

static bool isOpLoopInvariant(Operation &op, Value indVar, ValueRange iterArgs,
                              SmallPtrSetImpl<Operation *> &opsWithUsers,
                              SmallPtrSetImpl<Operation *> &opsToHoist) {
  LLVM_DEBUG(llvm::dbgs() << "\nIterating on op: " << op.getName() << "\n");

  if (auto ifOp = dyn_cast<IfOp>(op)) {
    if (!checkInvarianceOfNestedIfOp(ifOp, indVar, iterArgs, opsWithUsers,
                                    opsToHoist))
    return false;
  } else if (auto forOp = dyn_cast<ForOp>(op)) {
    if (!areAllOpsInTheBlockListInvariant(forOp.getLoopBody(), indVar, iterArgs,
                                          opsWithUsers, opsToHoist)) {
      return false;
    }
  } else if (auto parOp = dyn_cast<ParallelOp>(op)) {
    if (!areAllOpsInTheBlockListInvariant(parOp.getLoopBody(), indVar, iterArgs,
                                          opsWithUsers, opsToHoist)) {
      return false;
    }
  } else if (!isMemoryEffectFree(&op)) {
    return false;
  }

  // Check operands.
  for (unsigned int i = 0; i < op.getNumOperands(); ++i) {
    auto *operandSrc = op.getOperand(i).getDefiningOp();
    LLVM_DEBUG(llvm::dbgs() << "\t-> Iterating on operand at index: " << i << "\n");

    auto opr = op.getOperand(i);
    if (indVar == opr) {
      LLVM_DEBUG(llvm::dbgs() << "\t  -> index: " << i << " operand is Loop IV\n");
      return false;
    }
    if (llvm::is_contained(iterArgs, opr)) {
      LLVM_DEBUG(llvm::dbgs() << "\t  -> index: " << i <<  " operand is one of the iter_args\n");
      return false;
    }
    if (operandSrc) {
      LLVM_DEBUG(llvm::dbgs() << "\t  -> Iterating on operand src: " << operandSrc->getName() << "\n");
      if (opsWithUsers.count(operandSrc) && opsToHoist.count(operandSrc) == 0)
        return false;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "-> " << op.getName() << " is Invariant Op\n");
  opsToHoist.insert(&op);
  return true;
}

static bool
checkInvarianceOfNestedIfOp(IfOp ifOp, Value indVar, ValueRange iterArgs,
                           SmallPtrSetImpl<Operation *> &opsWithUsers,
                           SmallPtrSetImpl<Operation *> &opsToHoist) {
  if (!areAllOpsInTheBlockListInvariant(ifOp.getThenRegion(), indVar, iterArgs,
                                        opsWithUsers, opsToHoist))
    return false;

  if (!areAllOpsInTheBlockListInvariant(ifOp.getElseRegion(), indVar, iterArgs,
                                        opsWithUsers, opsToHoist))
    return false;
  
  return true;
}

void runOnForLoopOp(ForOp forOp) {
  auto *loopBody = forOp.getBody();
  auto indVar = forOp.getInductionVar();
  ValueRange iterArgs = forOp.getRegionIterArgs();

  SmallPtrSet<Operation *, 8> opsToHoist;
  SmallVector<Operation *, 8> opsToMove;
  SmallPtrSet<Operation *, 8> opsWithUsers;

  for (auto &op : *loopBody) {
    if (!op.use_empty())
      opsWithUsers.insert(&op);

    if (!isa<YieldOp>(op)) {
      if (isOpLoopInvariant(op, indVar, iterArgs, opsWithUsers, opsToHoist)) {
        opsToMove.push_back(&op);
      }
    }
  }

  for (auto *op: opsToMove)
    op->moveBefore(forOp);
  
  LLVM_DEBUG(forOp->print(llvm::dbgs() << "*** Modified loop ***\n"));
}

struct LoopInvariantCodeMotion
    : public impl::SCFLoopInvariantCodeMotionBase<LoopInvariantCodeMotion> { 
  void runOnOperation() override {
    getOperation()->walk([&](ForOp op) {
      LLVM_DEBUG(op->print(llvm::dbgs() << "\nInvariant Code Motion: Original loop\n"));
      runOnForLoopOp(op);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createSCFLoopInvariantCodeMotionPass() {
  return std::make_unique<LoopInvariantCodeMotion>();
}
