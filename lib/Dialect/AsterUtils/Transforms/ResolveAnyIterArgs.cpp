//===- ResolveAnyIterArgs.cpp - Resolve any-typed iter_args ---------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass specializes !aster_utils.any-typed scf.for iter_args to their
// concrete types by analyzing to_any/from_any chains. This is needed after
// struct destructuring of pipelined loops, where the value field of future
// structs becomes a bare any-typed iter_arg.
//
// The SCF pipeliner creates a rotation pattern where some iter_args are
// consumed (via from_any) and some are passed through unchanged in the yield.
// This pass handles both patterns:
//   - Consumed args: init is to_any, body uses from_any, yield is to_any
//   - Passthrough args: init is to_any, only use is yield, yield passes
//     the block arg to another candidate iter_arg position
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AsterUtils/IR/AsterUtilsDialect.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsOps.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsTypes.h"
#include "aster/Dialect/AsterUtils/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"

namespace mlir::aster {
namespace aster_utils {
#define GEN_PASS_DEF_RESOLVEANYITERARGS
#include "aster/Dialect/AsterUtils/Transforms/Passes.h.inc"
} // namespace aster_utils
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::aster_utils;

namespace {

/// Check if an scf.for has any !aster_utils.any-typed iter_args.
static bool hasAnyTypedIterArgs(scf::ForOp forOp) {
  for (auto arg : forOp.getInitArgs()) {
    if (isa<AnyTypeType>(arg.getType()))
      return true;
  }
  return false;
}

/// Resolve any-typed iter_args in a single scf.for op.
///
/// The analysis has three phases:
///   Phase 1: For each any-typed iter_arg, check that:
///     - Init is `to_any %x : T` (record candidate type T)
///     - All block arg uses are either `from_any ... : T` or yield passthrough
///     - All result uses are `from_any ... : T`
///   Phase 2: Validate yields. Each candidate's yield must be:
///     - `to_any %y : T` (new production), or
///     - A block arg of another candidate with the same type T (passthrough)
///     Iterate rejection until stable.
///   Phase 3: Build new ForOp with concrete types.
static LogicalResult resolveForOp(scf::ForOp forOp) {
  if (!hasAnyTypedIterArgs(forOp))
    return failure();

  unsigned numIterArgs = forOp.getInitArgs().size();
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());

  // Phase 1: For each any-typed iter_arg, check init, block arg uses, and
  // result uses. Record candidate concrete type.
  SmallVector<Type> candidateTypes(numIterArgs);
  for (unsigned i = 0; i < numIterArgs; ++i) {
    Value initArg = forOp.getInitArgs()[i];
    if (!isa<AnyTypeType>(initArg.getType()))
      continue;

    // Init must be to_any.
    auto toAnyInit = initArg.getDefiningOp<ToAnyOp>();
    if (!toAnyInit)
      continue;
    Type T = toAnyInit.getInput().getType();

    // All block arg uses must be from_any : T or yield passthrough.
    Value blockArg = forOp.getRegionIterArg(i);
    bool blockArgOk = true;
    for (OpOperand &use : blockArg.getUses()) {
      // Yield passthrough is fine - validated in Phase 2.
      if (isa<scf::YieldOp>(use.getOwner()))
        continue;
      auto fromAny = dyn_cast<FromAnyOp>(use.getOwner());
      if (!fromAny || fromAny.getResult().getType() != T) {
        blockArgOk = false;
        break;
      }
    }
    if (!blockArgOk)
      continue;

    // All result uses must be from_any : T.
    Value result = forOp.getResult(i);
    bool resultOk = true;
    for (OpOperand &use : result.getUses()) {
      auto fromAny = dyn_cast<FromAnyOp>(use.getOwner());
      if (!fromAny || fromAny.getResult().getType() != T) {
        resultOk = false;
        break;
      }
    }
    if (!resultOk)
      continue;

    candidateTypes[i] = T;
  }

  // Phase 2: Validate yields. Each candidate's yield value must be either:
  //   (a) to_any %x : T, or
  //   (b) a block arg of another candidate with the same type T
  // Rejecting one candidate may invalidate others (if they depend on it
  // through passthrough), so iterate until stable.
  bool changed = true;
  while (changed) {
    changed = false;
    for (unsigned i = 0; i < numIterArgs; ++i) {
      if (!candidateTypes[i])
        continue;

      Value yieldVal = yieldOp.getOperand(i);

      // Case (a): to_any with matching type.
      auto toAny = yieldVal.getDefiningOp<ToAnyOp>();
      if (toAny && toAny.getInput().getType() == candidateTypes[i])
        continue;

      // Case (b): block arg of this loop that is also a candidate with same T.
      auto blockArg = dyn_cast<BlockArgument>(yieldVal);
      if (blockArg && blockArg.getOwner() == forOp.getBody()) {
        unsigned argNum = blockArg.getArgNumber();
        // argNum > 0 means it's an iter_arg (0 is the IV).
        if (argNum > 0) {
          unsigned iterIdx = argNum - 1;
          if (candidateTypes[iterIdx] == candidateTypes[i])
            continue;
        }
      }

      // Neither case matched - reject this candidate.
      candidateTypes[i] = nullptr;
      changed = true;
    }
  }

  // Check if any candidates remain.
  bool hasSpecializable = false;
  for (unsigned i = 0; i < numIterArgs; ++i) {
    if (candidateTypes[i])
      hasSpecializable = true;
  }
  if (!hasSpecializable)
    return failure();

  // Phase 3: Build new ForOp with concrete types.
  OpBuilder builder(forOp);
  Location loc = forOp.getLoc();

  // Build new init args, unwrapping to_any for candidates.
  SmallVector<Value> newInitArgs;
  for (unsigned i = 0; i < numIterArgs; ++i) {
    if (candidateTypes[i]) {
      auto toAny = forOp.getInitArgs()[i].getDefiningOp<ToAnyOp>();
      newInitArgs.push_back(toAny.getInput());
    } else {
      newInitArgs.push_back(forOp.getInitArgs()[i]);
    }
  }

  // Create new ForOp with concrete-typed init args.
  auto newForOp =
      scf::ForOp::create(builder, loc, forOp.getLowerBound(),
                         forOp.getUpperBound(), forOp.getStep(), newInitArgs);

  Block *newBody = newForOp.getBody();
  Block *oldBody = forOp.getBody();

  // Remove auto-generated yield if present.
  if (newBody->mightHaveTerminator())
    newBody->getTerminator()->erase();

  // Set up mapping from old body values to new body values.
  builder.setInsertionPointToStart(newBody);
  IRMapping mapping;

  // Map induction variable.
  mapping.map(oldBody->getArgument(0), newBody->getArgument(0));

  // Map block args. For candidates: map old block arg to new concrete-typed
  // block arg (handles yield passthroughs), and map from_any results to new
  // block arg (handles consumption). Skip from_any ops during cloning.
  DenseSet<Operation *> skipOps;
  for (unsigned i = 0; i < numIterArgs; ++i) {
    Value oldBlockArg = oldBody->getArgument(i + 1);
    Value newBlockArg = newBody->getArgument(i + 1);

    // Always map old block arg to new block arg.
    mapping.map(oldBlockArg, newBlockArg);

    if (candidateTypes[i]) {
      // Additionally map from_any results to new block arg and skip them.
      for (OpOperand &use : oldBlockArg.getUses()) {
        if (isa<scf::YieldOp>(use.getOwner()))
          continue;
        auto fromAny = cast<FromAnyOp>(use.getOwner());
        mapping.map(fromAny.getResult(), newBlockArg);
        skipOps.insert(fromAny);
      }
    }
  }

  // Clone body ops (except skipped from_any ops and the terminator).
  for (auto &op : oldBody->without_terminator()) {
    if (skipOps.contains(&op))
      continue;
    builder.clone(op, mapping);
  }

  // Build new yield. For candidates: unwrap to_any if present, otherwise
  // the mapping handles passthrough block args automatically.
  SmallVector<Value> newYieldOperands;
  for (unsigned i = 0; i < numIterArgs; ++i) {
    Value yieldVal = yieldOp.getOperand(i);
    if (candidateTypes[i]) {
      // Try to unwrap to_any (new production case).
      auto toAny = yieldVal.getDefiningOp<ToAnyOp>();
      if (toAny) {
        newYieldOperands.push_back(mapping.lookupOrDefault(toAny.getInput()));
        continue;
      }
      // Passthrough: mapping converts old block arg to new concrete block arg.
    }
    newYieldOperands.push_back(mapping.lookupOrDefault(yieldVal));
  }
  scf::YieldOp::create(builder, loc, newYieldOperands);

  // Replace uses of old results.
  // For candidates: replace from_any users with the new concrete-typed result.
  // For non-candidates: direct replacement.
  builder.setInsertionPointAfter(newForOp);
  for (unsigned i = 0; i < numIterArgs; ++i) {
    Value oldResult = forOp.getResult(i);
    Value newResult = newForOp.getResult(i);

    if (candidateTypes[i]) {
      for (OpOperand &use : llvm::make_early_inc_range(oldResult.getUses())) {
        auto fromAny = cast<FromAnyOp>(use.getOwner());
        fromAny.getResult().replaceAllUsesWith(newResult);
        fromAny.erase();
      }
    } else {
      oldResult.replaceAllUsesWith(newResult);
    }
  }

  // Erase old ForOp.
  forOp.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// ResolveAnyIterArgs pass
//===----------------------------------------------------------------------===//

struct ResolveAnyIterArgs
    : public aster_utils::impl::ResolveAnyIterArgsBase<ResolveAnyIterArgs> {
public:
  using Base::Base;
  void runOnOperation() override;
};

} // namespace

void ResolveAnyIterArgs::runOnOperation() {
  // Collect all scf.for ops with any-typed iter_args. Process in reverse
  // order so inner loops are resolved before outer loops.
  SmallVector<scf::ForOp> worklist;
  getOperation()->walk([&](scf::ForOp forOp) {
    if (hasAnyTypedIterArgs(forOp))
      worklist.push_back(forOp);
  });

  for (auto forOp : llvm::reverse(worklist))
    (void)resolveForOp(forOp);
}
