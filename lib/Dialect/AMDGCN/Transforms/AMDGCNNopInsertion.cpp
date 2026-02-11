//===- AMDGCNNopInsertion.cpp - NOP Insertion Pass ------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Interfaces/RegisterType.h"

#include "aster/Interfaces/VerifierAttr.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>

// Include generated instruction definitions
#include "aster/Dialect/AMDGCN/IR/AMDGCNInsts.h.inc"
#include "mlir/Support/WalkResult.h"

#define DEBUG_TYPE "amdgcn-nop-insertion"

namespace mlir {
namespace aster {
namespace amdgcn {
#define GEN_PASS_DEF_AMDGCNNOPINSERTION
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace aster
} // namespace mlir

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Check if a register range equals another register range.
/// Returns true if the ranges are the same.
static bool registerRangesEqual(RegisterRange range1, RegisterRange range2) {
  return range1.begin().getRegister() == range2.begin().getRegister() &&
         range1.size() == range2.size();
}

/// Check if a register range overlaps with another register range.
/// Returns true if there's any overlap.
static bool registerRangesOverlap(RegisterRange range1, RegisterRange range2) {
  // Check if ranges overlap: [start1, end1) overlaps [start2, end2)
  // where end = start + size
  int start1 = range1.begin().getRegister();
  int end1 = start1 + range1.size();
  int start2 = range2.begin().getRegister();
  int end2 = start2 + range2.size();

  return !(end1 <= start2 || end2 <= start1);
}

/// Get the VGPR range from a value if it's a VGPR range type.
static std::optional<RegisterRange> getVGPRRange(Value value) {
  Type type = value.getType();
  auto regType = dyn_cast<AMDGCNRegisterTypeInterface>(type);
  if (!regType)
    return std::nullopt;

  if (regType.getRegisterKind() != RegisterKind::VGPR)
    return std::nullopt;

  return regType.getAsRange();
}

/// Check if an operation writes to VGPRs that overlap with the given range.
static bool writesToVGPRRange(Operation *op, RegisterRange targetRange) {
  // Check if this operation implements AMDGCNInstOpInterface
  auto instOp = dyn_cast<AMDGCNInstOpInterface>(op);
  if (!instOp)
    return false;

  // Check all output operands (results)
  for (OpOperand &operand : instOp.getInstOutsMutable()) {
    Value result = operand.get();
    auto resultRange = getVGPRRange(result);
    if (resultRange && registerRangesOverlap(*resultRange, targetRange))
      return true;
  }

  return false;
}

/// Check if an operation is a VALU (Vector ALU) instruction.
/// VALU instructions include VOP1, VOP2, VOP3, VOP3P operations.
/// TODO: something less brittle in tablegen directly.
static bool isVALUInstruction(Operation *op) {
  StringRef opName = op->getName().getStringRef();
  // Check if operation name starts with VOP prefixes
  return opName.starts_with("amdgcn.vop1") ||
         opName.starts_with("amdgcn.vop2") ||
         opName.starts_with("amdgcn.vop3") ||
         opName.starts_with("amdgcn.vop3p");
}

/// Check if an operation is a non-MFMA VALU instruction.
static bool isNonMFMAVALUInstruction(Operation *op) {
  return isVALUInstruction(op) && !isa<inst::VOP3PMAIOp>(op);
}

/// Get the number of passes for an MFMA opcode.
/// Pass count determines hazard wait state requirements per ISA Table 37.
/// Each pass is 4 clock cycles.
static int getNumPasses(OpCode opcode) {
  switch (opcode) {
  // 2-pass XDL instructions (16x16x16 variants)
  case OpCode::V_MFMA_F32_16X16X16_F16:
  case OpCode::V_MFMA_F32_16X16X16_BF16:
  case OpCode::V_MFMA_F16_16X16X16_F16:
    return 2;
  default:
    llvm_unreachable("Unknown MFMA opcode for pass count");
  }
}

/// Look up pass-dependent NOP count from ISA Table 37.
/// Tables provide wait counts for 2-pass, 4-pass, 8-pass, 16-pass variants.
static int getPassDependentNops(int numPasses, int nop2, int nop4, int nop8,
                                int nop16) {
  switch (numPasses) {
  case 2:
    return nop2;
  case 4:
    return nop4;
  case 8:
    return nop8;
  case 16:
    return nop16;
  default:
    llvm_unreachable("Invalid MFMA pass count");
  }
}

/// Get the SGPR range from a value if it's an SGPR range type.
static std::optional<RegisterRange> getSGPRRange(Value value) {
  Type type = value.getType();
  auto regType = dyn_cast<AMDGCNRegisterTypeInterface>(type);
  if (!regType)
    return std::nullopt;

  if (regType.getRegisterKind() != RegisterKind::SGPR)
    return std::nullopt;

  return regType.getAsRange();
}

/// Check if an operation is a VMEM (Vector Memory) instruction.
/// VMEM instructions include MUBUF, MTBUF, and FLAT memory operations.
/// TODO: something less brittle in tablegen directly.
static bool isVMEMInstruction(Operation *op) {
  StringRef opName = op->getName().getStringRef();
  // Check if operation name starts with VMEM prefixes
  if (opName.starts_with("amdgcn.mubuf.") ||
      opName.starts_with("amdgcn.mtbuf."))
    return true;
  // FLAT operations that are memory operations (load/store)
  if (auto loadOp = dyn_cast<LoadOp>(op);
      loadOp && loadOp.getInstKind() == MemoryInstructionKind::Flat)
    return true;
  if (auto storeOp = dyn_cast<StoreOp>(op);
      storeOp && storeOp.getInstKind() == MemoryInstructionKind::Flat)
    return true;
  return false;
}

/// Check if a VMEM instruction reads from SGPRs that overlap with the given
/// range.

static bool vMemReadsFromSGPRRange(Operation *op, RegisterRange targetRange) {
  assert(isVMEMInstruction(op) && "Expected VMEM instruction");
  auto instOp = cast<AMDGCNInstOpInterface>(op);

  // TODO: more ops here. Also, use a MemoryOpInterface.

  // For GlobalLoadOp and GlobalStoreOp, check the address operand directly
  if (auto loadOp = dyn_cast<LoadOp>(op);
      loadOp && loadOp.getInstKind() == MemoryInstructionKind::Flat) {
    Value addr = loadOp.getAddr();
    auto addrRange = getSGPRRange(addr);
    if (addrRange && registerRangesOverlap(*addrRange, targetRange))
      return true;
    return false;
  }
  if (auto storeOp = dyn_cast<StoreOp>(op);
      storeOp && storeOp.getInstKind() == MemoryInstructionKind::Flat) {
    Value addr = storeOp.getAddr();
    auto addrRange = getSGPRRange(addr);
    if (addrRange && registerRangesOverlap(*addrRange, targetRange))
      return true;
    return false;
  }

  // For other VMEM instructions, check all input operands for SGPR addresses.
  for (OpOperand &operand : instOp.getInstInsMutable()) {
    Value input = operand.get();
    auto inputRange = getSGPRRange(input);
    if (inputRange && registerRangesOverlap(*inputRange, targetRange))
      return true;
  }

  return false;
}

//===----------------------------------------------------------------------===//
// NOP insertion case structure
//===----------------------------------------------------------------------===//

/// Enum to specify which type of NOP to use
enum class NopType { SNOP, VNOP, BOTH };

/// Represents a case where a NOP needs to be inserted.
struct NopInsertionCase {
  Operation *inst1;   // First instruction (e.g., FLAT_STORE_X3/X4)
  Operation *inst2;   // Second instruction (e.g., VALU write to same VGPRs)
  int lookaheadCount; // How many instructions ahead inst2 was found
  int valuCount;      // How many VALU instructions between inst1 and inst2
  int requiredNops;   // Number of NOPs required
  int caseNumber;     // Case number from Table 11 (e.g., 8)
  NopType nopType;    // Type of NOP to insert (SNOP, VNOP, BOTH)

  NopInsertionCase(Operation *i1, Operation *i2, int lookahead, int valu,
                   int nops, int caseNum = -1, NopType nop = NopType::BOTH)
      : inst1(i1), inst2(i2), lookaheadCount(lookahead), valuCount(valu),
        requiredNops(nops), caseNumber(caseNum), nopType(nop) {}

  // Comparison operators for SetVector deduplication
  bool operator==(const NopInsertionCase &other) const {
    return inst1 == other.inst1 && inst2 == other.inst2 &&
           caseNumber == other.caseNumber && nopType == other.nopType;
  }
  bool operator<(const NopInsertionCase &other) const {
    if (inst1 != other.inst1)
      return inst1 < other.inst1;
    if (inst2 != other.inst2)
      return inst2 < other.inst2;
    if (caseNumber != other.caseNumber)
      return caseNumber < other.caseNumber;
    return nopType < other.nopType;
  }
};

// DenseMapInfo specialization for NopInsertionCase (required by SetVector)
namespace llvm {
template <>
struct DenseMapInfo<NopInsertionCase> {
  static inline NopInsertionCase getEmptyKey() {
    return NopInsertionCase(DenseMapInfo<Operation *>::getEmptyKey(),
                            DenseMapInfo<Operation *>::getEmptyKey(), 0, 0, 0,
                            0, NopType::SNOP);
  }
  static inline NopInsertionCase getTombstoneKey() {
    return NopInsertionCase(DenseMapInfo<Operation *>::getTombstoneKey(),
                            DenseMapInfo<Operation *>::getTombstoneKey(), 0, 0,
                            0, 0, NopType::SNOP);
  }
  static unsigned getHashValue(const NopInsertionCase &val) {
    return hash_combine(val.inst1, val.inst2, val.caseNumber,
                        static_cast<int>(val.nopType));
  }
  static bool isEqual(const NopInsertionCase &lhs,
                      const NopInsertionCase &rhs) {
    return lhs.inst1 == rhs.inst1 && lhs.inst2 == rhs.inst2 &&
           lhs.caseNumber == rhs.caseNumber && lhs.nopType == rhs.nopType;
  }
};
} // namespace llvm

/// Structure holding the predicates and dependency checker for a NOP insertion
/// case.
struct NopInsertionCaseDef {
  std::function<bool(Operation *)> selectInst1;
  std::function<bool(Operation *)> selectInst2;
  std::function<int(Operation *, Operation *)> checkDependency;
  int maxLookaheadCount; // How many instructions ahead inst2 was found
  int caseNumber;        // Case number from Table 11 (e.g., 8)
  NopType nopType;       // Type of NOP to insert (SNOP or VNOP)

  NopInsertionCaseDef(std::function<bool(Operation *)> s1,
                      std::function<bool(Operation *)> s2,
                      std::function<int(Operation *, Operation *)> check,
                      int maxLookaheadCount, int caseNum = -1,
                      NopType nop = NopType::BOTH)
      : selectInst1(s1), selectInst2(s2), checkDependency(check),
        maxLookaheadCount(maxLookaheadCount), caseNumber(caseNum),
        nopType(nop) {}
};

/// Find NOP insertion cases by scanning a block.
///
/// This function looks for pairs of instructions (inst1, inst2) where:
/// - inst1 matches selectInst1 predicate
/// - inst2 matches selectInst2 predicate and appears within maxLookahead
/// - A dependency exists between inst1 and inst2 that requires NOPs
///
/// \param block The block to scan
/// \param caseDef The case definition containing selectInst1, selectInst2, and
/// checkDependency \param maxLookahead Maximum number of instructions to look
/// ahead \returns Vector of NOP insertion cases found
static SmallVector<NopInsertionCase>
findNopInsertionCases(Block *block, const NopInsertionCaseDef &caseDef,
                      int maxLookahead) {
  SmallVector<NopInsertionCase> cases;

  for (auto it = block->begin(); it != block->end(); ++it) {
    Operation *currentOp = &*it;

    // Check if this matches inst1
    if (!caseDef.selectInst1(currentOp))
      continue;

    // Look ahead for inst2
    auto nextIt = std::next(it);
    int lookaheadCount = 0;
    int valuCount = 0;

    for (; nextIt != block->end() && lookaheadCount < maxLookahead; ++nextIt) {
      Operation *nextOp = &*nextIt;

      // Only consider AMDGCN instruction operations
      if (!isa<AMDGCNInstOpInterface>(nextOp))
        continue;

      ++lookaheadCount;
      // Track VALU instructions separately - only these count as wait states
      // for MFMA hazards (ISA manual: "NOPs or independent VALU instructions").
      if (isVALUInstruction(nextOp))
        ++valuCount;

      // Check if this matches inst2
      if (!caseDef.selectInst2(nextOp))
        continue;

      // Check if there's a dependency requiring NOPs
      int requiredNops = caseDef.checkDependency(currentOp, nextOp);
      if (requiredNops > 0) {
        // valuCount between inst1 and inst2, excluding inst2 itself.
        int valuBetween = isVALUInstruction(nextOp) ? valuCount - 1 : valuCount;
        cases.emplace_back(currentOp, nextOp, lookaheadCount, valuBetween,
                           requiredNops, caseDef.caseNumber, caseDef.nopType);
        // Only record first match per inst1 (can be extended if needed)
        break;
      }
    }
  }

  return cases;
}

//===----------------------------------------------------------------------===//
// Case definitions
//===----------------------------------------------------------------------===//

/// Case 8: FLAT_STORE_X3/X4 -> Write VGPRs holding writedata
/// Table 11, Case 8: Requires 1 NOP
static NopInsertionCaseDef getCase8Definition() {
  auto selectInst1 = [](Operation *op) -> bool {
    auto globalStore = dyn_cast<StoreOp>(op);
    if (!globalStore ||
        globalStore.getInstKind() != MemoryInstructionKind::Flat)
      return false;
    OpCode opcode = globalStore.getOpcode();
    return opcode == OpCode::GLOBAL_STORE_DWORDX3 ||
           opcode == OpCode::GLOBAL_STORE_DWORDX4;
  };

  auto selectInst2 = [](Operation *op) -> bool {
    // inst2 is any operation that writes to VGPRs
    auto instOp = dyn_cast<AMDGCNInstOpInterface>(op);
    if (!instOp)
      return false;
    // Check if it has any VGPR outputs
    for (OpOperand &operand : instOp.getInstOutsMutable()) {
      if (getVGPRRange(operand.get()))
        return true;
    }
    return false;
  };

  auto checkDependency = [](Operation *inst1, Operation *inst2) -> int {
    auto globalStore = cast<StoreOp>(inst1);
    Value dataValue = globalStore.getData();
    auto dataRange = getVGPRRange(dataValue);
    if (!dataRange) {
      LDBG() << "Store op " << *inst1
             << " does not have VGPR range data operand";
      return 0;
    }

    LDBG() << "Checking dependency: store " << *inst1 << " with data range: ["
           << dataRange->begin().getRegister() << ", "
           << dataRange->begin().getRegister() + dataRange->size() << ")";

    // Check if inst2 writes to VGPRs that overlap with the store's data
    if (writesToVGPRRange(inst2, *dataRange)) {
      LDBG() << "Found dependency: " << *inst2
             << " writes to VGPRs overlapping with store data range";
      return 1; // Case 8 requires 1 NOP
    }

    return 0;
  };

  return NopInsertionCaseDef(selectInst1, selectInst2, checkDependency,
                             /*maxLookaheadCount=*/10, /*caseNumber=*/8,
                             /*nopType=*/NopType::VNOP);
}

/// Case 9: FLAT_STORE_X3/X4 -> VALU writes VGPRs holding writedata
/// Table 11, Case 9: Requires 2 NOPs
/// Similar to Case 8, but inst2 must be a VALU instruction
static NopInsertionCaseDef getCase9Definition() {
  auto selectInst1 = [](Operation *op) -> bool {
    auto globalStore = dyn_cast<StoreOp>(op);
    if (!globalStore ||
        globalStore.getInstKind() != MemoryInstructionKind::Flat)
      return false;
    OpCode opcode = globalStore.getOpcode();
    return opcode == OpCode::GLOBAL_STORE_DWORDX3 ||
           opcode == OpCode::GLOBAL_STORE_DWORDX4;
  };

  auto selectInst2 = [](Operation *op) -> bool {
    // inst2 must be a VALU instruction that writes to VGPRs
    if (!isVALUInstruction(op))
      return false;
    auto instOp = dyn_cast<AMDGCNInstOpInterface>(op);
    if (!instOp)
      return false;
    // Check if it has any VGPR outputs
    for (OpOperand &operand : instOp.getInstOutsMutable()) {
      if (getVGPRRange(operand.get()))
        return true;
    }
    return false;
  };

  auto checkDependency = [](Operation *inst1, Operation *inst2) -> int {
    auto globalStore = cast<StoreOp>(inst1);
    Value dataValue = globalStore.getData();
    auto dataRange = getVGPRRange(dataValue);
    if (!dataRange) {
      LDBG() << "Store op " << *inst1
             << " does not have VGPR range data operand";
      return 0;
    }

    LDBG() << "Checking dependency (Case 9): store " << *inst1
           << " with data range: [" << dataRange->begin().getRegister() << ", "
           << dataRange->begin().getRegister() + dataRange->size() << ")";

    // Check if inst2 (VALU) writes to VGPRs that overlap with the store's data
    if (writesToVGPRRange(inst2, *dataRange)) {
      LDBG() << "Found dependency: " << *inst2
             << " (VALU) writes to VGPRs overlapping with store data range";
      return 2; // Case 9 requires 2 NOPs
    }

    return 0;
  };

  return NopInsertionCaseDef(selectInst1, selectInst2, checkDependency,
                             /*maxLookaheadCount=*/10, /*caseNumber=*/9,
                             /*nopType=*/NopType::VNOP);
}

/// Case 10: VALU writes SGPR -> VMEM reads that SGPR
/// Table 11, Case 10: Requires 5 NOPs
static NopInsertionCaseDef getCase10Definition() {
  auto selectInst1 = [](Operation *op) -> bool {
    auto instOp = dyn_cast<AMDGCNInstOpInterface>(op);
    if (!instOp || !isVALUInstruction(op))
      return false;
    // Check if it has any SGPR outputs
    for (OpOperand &operand : instOp.getInstOutsMutable()) {
      if (getSGPRRange(operand.get()))
        return true;
    }
    return false;
  };

  auto selectInst2 = [](Operation *op) -> bool {
    return isVMEMInstruction(op);
  };

  auto checkDependency = [](Operation *inst1, Operation *inst2) -> int {
    // Check all SGPR outputs from inst1 (VALU)
    auto inst1Op = cast<AMDGCNInstOpInterface>(inst1);
    for (OpOperand &operand : inst1Op.getInstOutsMutable()) {
      Value sgprValue = operand.get();
      auto sgprRange = getSGPRRange(sgprValue);
      if (!sgprRange)
        continue;

      LDBG() << "Checking dependency (Case 10): VALU " << *inst1
             << " writes SGPR range: [" << sgprRange->begin().getRegister()
             << ", " << sgprRange->begin().getRegister() + sgprRange->size()
             << ")";

      // Check if inst2 (VMEM) reads from SGPRs that overlap with inst1's SGPR
      // output
      if (vMemReadsFromSGPRRange(inst2, *sgprRange)) {
        LDBG() << "Found dependency: " << *inst2
               << " (VMEM) reads from SGPRs overlapping with VALU output range";
        return 5; // Case 10 requires 5 NOPs
      }
    }

    return 0;
  };

  return NopInsertionCaseDef(selectInst1, selectInst2, checkDependency,
                             /*maxLookaheadCount=*/10, /*caseNumber=*/10,
                             /*nopType=*/NopType::SNOP);
}

/// Case 100: Non-DLops VALU Write VGPR -> V_MFMA* read VGPR
/// Table 37, Case 100: Requires 2 NOPs
/// "No internal 4 & 8 cycle forwarding path."
static NopInsertionCaseDef getCase100Definition() {
  auto selectInst1 = [](Operation *op) -> bool {
    return isNonMFMAVALUInstruction(op);
  };

  auto selectInst2 = [](Operation *op) -> bool {
    return isa<inst::VOP3PMAIOp>(op);
  };

  auto checkDependency = [](Operation *inst1, Operation *inst2) -> int {
    auto instOp1 = cast<AMDGCNInstOpInterface>(inst1);
    auto maiOp2 = cast<inst::VOP3PMAIOp>(inst2);

    // Check all VGPR outputs from the VALU instruction.
    for (OpOperand &outOp : instOp1.getInstOutsMutable()) {
      auto outRange = getVGPRRange(outOp.get());
      if (!outRange)
        continue;

      // Check if MFMA reads any of those VGPRs as SrcA, SrcB, or SrcC.
      Value inputs[] = {maiOp2.getA(), maiOp2.getB(), maiOp2.getC()};
      for (Value input : inputs) {
        auto inRange = getVGPRRange(input);
        if (inRange && registerRangesOverlap(*outRange, *inRange))
          return 2;
      }
    }
    return 0;
  };
  return NopInsertionCaseDef(selectInst1, selectInst2, checkDependency,
                             /*maxLookaheadCount=*/4, /*caseNumber=*/100,
                             /*nopType=*/NopType::VNOP);
}

/// Case 101: DL ops Write VGPR -> DLops read VGPR
/// Table 37, Case 101 has three sub-cases:
///   (a) SrcC same opcode: 0 NOPs (back-to-back SrcC forwarding)
///   (b) SrcA/B same opcode: 3 NOPs (no SrcA/B forwarding in DLops)
///   (c) Different opcode read/write VGPR (RAW+WAW): 3 NOPs
/// Note: For XDL ops, Cases 102-106 provide more specific rules that override
/// the generic Case 101 (a) rule. Case 101 (b) and (c) still apply.
static NopInsertionCaseDef getCase101Definition() {
  auto selectInst1 = [](Operation *op) -> bool {
    return isa<inst::VOP3PMAIOp>(op);
  };

  auto selectInst2 = [](Operation *op) -> bool {
    return isa<inst::VOP3PMAIOp>(op);
  };

  auto checkDependency = [](Operation *inst1, Operation *inst2) -> int {
    auto maiOp1 = cast<inst::VOP3PMAIOp>(inst1);
    auto maiOp2 = cast<inst::VOP3PMAIOp>(inst2);

    Value vdst1 = maiOp1.getVdst();
    auto vdst1Range = getVGPRRange(vdst1);
    assert(vdst1Range && "VGPR range should be valid");

    // Same opcode sub-cases.
    if (maiOp1.getOpcode() == maiOp2.getOpcode()) {
      // Sub-case (a): SrcC forwarding - 0 NOPs.
      // Note: for XDL ops, Case 102 overrides this with pass-dependent waits.
      // This case definition is kept for DLops generality, but Case 102 is
      // registered separately and takes precedence for XDL via max-of-all.
      Value srcC = maiOp2.getC();
      auto srcCRange = getVGPRRange(srcC);
      assert(srcCRange && "VGPR range should be valid");
      if (registerRangesEqual(*vdst1Range, *srcCRange))
        return 0;

      // Sub-case (b): SrcA or SrcB read - 3 NOPs.
      for (Value input : {maiOp2.getA(), maiOp2.getB()}) {
        auto inRange = getVGPRRange(input);
        if (inRange && registerRangesOverlap(*vdst1Range, *inRange))
          return 3;
      }

      // SrcC overlapped but not exact (still same opcode) - handled by
      // Case 103 which provides more specific pass-dependent values.
      if (registerRangesOverlap(*vdst1Range, *srcCRange))
        return 3;
    }

    // Sub-case (c): Different opcode - any read/write VGPR overlap is 3 NOPs.
    if (maiOp1.getOpcode() != maiOp2.getOpcode()) {
      // Check all inputs and outputs of inst2 for overlap with vdst1.
      auto op2 = cast<AMDGCNInstOpInterface>(inst2);
      for (Value operand : op2.getInstIns()) {
        auto rTy = dyn_cast<RegisterTypeInterface>(operand.getType());
        if (rTy && registerRangesOverlap(*vdst1Range, rTy.getAsRange()))
          return 3;
      }
      for (Value operand : op2.getInstOuts()) {
        auto rTy = dyn_cast<RegisterTypeInterface>(operand.getType());
        if (rTy && registerRangesOverlap(*vdst1Range, rTy.getAsRange()))
          return 3;
      }
    }

    return 0;
  };
  return NopInsertionCaseDef(selectInst1, selectInst2, checkDependency,
                             /*maxLookaheadCount=*/5, /*caseNumber=*/101,
                             /*nopType=*/NopType::VNOP);
}

/// Case 102: XDL Write VGPR -> XDL read VGPR as SrcC exactly same vDst
/// Table 37, Case 102: Pass-dependent waits.
///   2-pass: 2, 4-pass: 0, 8-pass: 0, 16-pass: 0
/// "The two V_MFMA must be the same number passes and vDst and vSrc start from
/// the same offset and same VGPR size."
static NopInsertionCaseDef getCase102Definition() {
  auto selectInst1 = [](Operation *op) -> bool {
    return isa<inst::VOP3PMAIOp>(op);
  };

  auto selectInst2 = [](Operation *op) -> bool {
    return isa<inst::VOP3PMAIOp>(op);
  };

  auto checkDependency = [](Operation *inst1, Operation *inst2) -> int {
    auto maiOp1 = cast<inst::VOP3PMAIOp>(inst1);
    auto maiOp2 = cast<inst::VOP3PMAIOp>(inst2);

    Value vdst1 = maiOp1.getVdst();
    auto vdst1Range = getVGPRRange(vdst1);
    assert(vdst1Range && "VGPR range should be valid");

    Value srcC = maiOp2.getC();
    auto srcCRange = getVGPRRange(srcC);
    assert(srcCRange && "VGPR range should be valid");

    // Must be same pass count, exact same vDst/SrcC range.
    int passes1 = getNumPasses(maiOp1.getOpcode());
    int passes2 = getNumPasses(maiOp2.getOpcode());
    if (passes1 == passes2 && registerRangesEqual(*vdst1Range, *srcCRange))
      return getPassDependentNops(passes1, /*2p=*/2, /*4p=*/0, /*8p=*/0,
                                  /*16p=*/0);

    return 0;
  };
  return NopInsertionCaseDef(selectInst1, selectInst2, checkDependency,
                             /*maxLookaheadCount=*/4, /*caseNumber=*/102,
                             /*nopType=*/NopType::VNOP);
}

/// Case 103: XDL Write VGPR -> XDL read VGPR as SrcC overlapped with vDst
/// Table 37, Case 103: Pass-dependent waits.
///   2-pass: 3, 4-pass: 5, 8-pass: 9, 16-pass: 17
static NopInsertionCaseDef getCase103Definition() {
  auto selectInst1 = [](Operation *op) -> bool {
    return isa<inst::VOP3PMAIOp>(op);
  };

  auto selectInst2 = [](Operation *op) -> bool {
    return isa<inst::VOP3PMAIOp>(op);
  };

  auto checkDependency = [](Operation *inst1, Operation *inst2) -> int {
    auto maiOp1 = cast<inst::VOP3PMAIOp>(inst1);
    auto maiOp2 = cast<inst::VOP3PMAIOp>(inst2);

    Value vdst1 = maiOp1.getVdst();
    auto vdst1Range = getVGPRRange(vdst1);
    assert(vdst1Range && "VGPR range should be valid");

    Value srcC = maiOp2.getC();
    auto srcCRange = getVGPRRange(srcC);
    assert(srcCRange && "VGPR range should be valid");

    // Overlapped but NOT exact same (exact same is Case 102).
    if (registerRangesOverlap(*vdst1Range, *srcCRange) &&
        !registerRangesEqual(*vdst1Range, *srcCRange)) {
      int passes1 = getNumPasses(maiOp1.getOpcode());
      return getPassDependentNops(passes1, /*2p=*/3, /*4p=*/5, /*8p=*/9,
                                  /*16p=*/17);
    }

    return 0;
  };
  return NopInsertionCaseDef(selectInst1, selectInst2, checkDependency,
                             /*maxLookaheadCount=*/19, /*caseNumber=*/103,
                             /*nopType=*/NopType::VNOP);
}

/// Case 105: XDL Write VGPR -> V_MFMA read VGPR as SrcA or SrcB
/// Table 37, Case 105: Pass-dependent waits.
///   2-pass: 5, 4-pass: 7, 8-pass: 11, 16-pass: 19
/// "No internal forwarding path waits for previous V_MFMA commit result to
/// VGPR."
static NopInsertionCaseDef getCase105Definition() {
  auto selectInst1 = [](Operation *op) -> bool {
    return isa<inst::VOP3PMAIOp>(op);
  };

  auto selectInst2 = [](Operation *op) -> bool {
    return isa<inst::VOP3PMAIOp>(op);
  };

  auto checkDependency = [](Operation *inst1, Operation *inst2) -> int {
    auto maiOp1 = cast<inst::VOP3PMAIOp>(inst1);
    auto maiOp2 = cast<inst::VOP3PMAIOp>(inst2);

    Value vdst1 = maiOp1.getVdst();
    auto vdst1Range = getVGPRRange(vdst1);
    assert(vdst1Range && "VGPR range should be valid");

    // Check SrcA and SrcB of the second MFMA.
    for (Value input : {maiOp2.getA(), maiOp2.getB()}) {
      auto inRange = getVGPRRange(input);
      if (inRange && registerRangesOverlap(*vdst1Range, *inRange)) {
        int passes1 = getNumPasses(maiOp1.getOpcode());
        return getPassDependentNops(passes1, /*2p=*/5, /*4p=*/7, /*8p=*/11,
                                    /*16p=*/19);
      }
    }

    return 0;
  };
  return NopInsertionCaseDef(selectInst1, selectInst2, checkDependency,
                             /*maxLookaheadCount=*/21, /*caseNumber=*/105,
                             /*nopType=*/NopType::VNOP);
}

/// Case 106: XDL Write VGPR -> VM/LDS/FLAT/VALU read/write VGPR overlapped
/// with vDst
/// Table 37, Case 106: Pass-dependent waits.
///   2-pass: 5, 4-pass: 7, 8-pass: 11, 16-pass: 19
static NopInsertionCaseDef getCase106Definition() {
  auto selectInst1 = [](Operation *op) -> bool {
    return isa<inst::VOP3PMAIOp>(op);
  };

  auto selectInst2 = [](Operation *op) -> bool {
    // VM, LDS, FLAT, or non-MFMA VALU instructions.
    // Exclude MFMA-to-MFMA which is handled by Cases 101-105.
    if (isa<inst::VOP3PMAIOp>(op))
      return false;
    return isa<StoreOp, LoadOp>(op) || isNonMFMAVALUInstruction(op);
  };

  auto checkDependency = [](Operation *inst1, Operation *inst2) -> int {
    auto maiOp1 = cast<inst::VOP3PMAIOp>(inst1);
    auto op2 = cast<AMDGCNInstOpInterface>(inst2);

    Value vdst1 = maiOp1.getVdst();
    auto vdst1Range = getVGPRRange(vdst1);
    assert(vdst1Range && "VGPR range should be valid");

    int passes1 = getNumPasses(maiOp1.getOpcode());
    int numNops = getPassDependentNops(passes1, /*2p=*/5, /*4p=*/7, /*8p=*/11,
                                       /*16p=*/19);

    // Check if inst2 reads or writes VGPRs overlapping with MFMA vDst.
    for (Value operand : op2.getInstIns()) {
      auto rTy = dyn_cast<RegisterTypeInterface>(operand.getType());
      if (!rTy)
        continue;
      if (registerRangesOverlap(*vdst1Range, rTy.getAsRange()))
        return numNops;
    }
    for (Value operand : op2.getInstOuts()) {
      auto rTy = dyn_cast<RegisterTypeInterface>(operand.getType());
      if (!rTy)
        continue;
      if (registerRangesOverlap(*vdst1Range, rTy.getAsRange()))
        return numNops;
    }
    return 0;
  };
  return NopInsertionCaseDef(selectInst1, selectInst2, checkDependency,
                             /*maxLookaheadCount=*/21, /*caseNumber=*/106,
                             /*nopType=*/NopType::VNOP);
}

/// Case ConservativeExtraDelays: Conservative extra delays after MFMA,
/// global_load, ds_read Inserts NOPs unconditionally after these operations
/// for debugging.
static NopInsertionCaseDef
getCaseConservativeExtraDelaysDefinition(unsigned numNops) {
  auto selectInst1 = [](Operation *op) -> bool {
    return isa<inst::VOP3PMAIOp, LoadOp, StoreOp>(op);
  };

  auto selectInst2 = [](Operation *op) -> bool { return true; };

  auto checkDependency = [numNops](Operation *inst1, Operation *inst2) -> int {
    (void)inst1;
    (void)inst2;
    // Conservatively always insert NOPs
    return numNops;
  };

  return NopInsertionCaseDef(selectInst1, selectInst2, checkDependency,
                             /*maxLookaheadCount=*/1);
}

//===----------------------------------------------------------------------===//
// AMDGCNNopInsertion pass
//===----------------------------------------------------------------------===//
namespace {
struct AMDGCNNopInsertion
    : public amdgcn::impl::AMDGCNNopInsertionBase<AMDGCNNopInsertion> {
public:
  using Base::Base;

  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// AMDGCNNopInsertion pass implementation
//===----------------------------------------------------------------------===//
void AMDGCNNopInsertion::runOnOperation() {
  Operation *op = getOperation();

  if (failed(runVerifiersOnOp<IsAllocatedOpAttr>(op, getAnalysisManager())))
    return signalPassFailure();

  // Use SetVector to avoid duplicate entries
  SetVector<NopInsertionCase> allCases;

  // Walk through all blocks in the operation
  op->walk([&](Block *block) {
    // TODO: cases 1 - 7

    // Case 8: FLAT_STORE_X3/X4 -> Write VGPRs holding writedata (1 NOP)
    auto case8Def = getCase8Definition();
    auto cases8 =
        findNopInsertionCases(block, case8Def, case8Def.maxLookaheadCount);
    for (const auto &case8 : cases8) {
      allCases.insert(case8);
    }

    // Case 9: FLAT_STORE_X3/X4 -> VALU writes VGPRs holding writedata (2 NOPs)
    auto case9Def = getCase9Definition();
    auto cases9 =
        findNopInsertionCases(block, case9Def, case9Def.maxLookaheadCount);
    for (const auto &case9 : cases9) {
      allCases.insert(case9);
    }

    // Case 10: VALU writes SGPR -> VMEM reads that SGPR (5 NOPs)
    auto case10Def = getCase10Definition();
    auto cases10 =
        findNopInsertionCases(block, case10Def, case10Def.maxLookaheadCount);
    for (const auto &case10 : cases10) {
      allCases.insert(case10);
    }

    // TODO: cases 11 - 21

    // Case 100: Non-DLops VALU Write VGPR -> MFMA reads VGPR (2 NOPs)
    auto case100Def = getCase100Definition();
    auto cases100 =
        findNopInsertionCases(block, case100Def, case100Def.maxLookaheadCount);
    for (const auto &c : cases100)
      allCases.insert(c);

    // Case 101: DL ops Write VGPR -> DLops read VGPR (0-3 NOPs depending on
    // sub-case: SrcC same opcode = 0, SrcA/B same opcode = 3, different
    // opcode = 3)
    auto case101Def = getCase101Definition();
    auto cases101 =
        findNopInsertionCases(block, case101Def, case101Def.maxLookaheadCount);
    for (const auto &c : cases101)
      allCases.insert(c);

    // Case 102: XDL Write VGPR -> XDL read SrcC exact same vDst (2 NOPs for
    // 2-pass, 0 for 4+)
    auto case102Def = getCase102Definition();
    auto cases102 =
        findNopInsertionCases(block, case102Def, case102Def.maxLookaheadCount);
    for (const auto &c : cases102)
      allCases.insert(c);

    // Case 103: XDL Write VGPR -> XDL read SrcC overlapped (3 NOPs for
    // 2-pass)
    auto case103Def = getCase103Definition();
    auto cases103 =
        findNopInsertionCases(block, case103Def, case103Def.maxLookaheadCount);
    for (const auto &c : cases103)
      allCases.insert(c);

    // Case 105: XDL Write VGPR -> MFMA read SrcA/B (5 NOPs for 2-pass)
    auto case105Def = getCase105Definition();
    auto cases105 =
        findNopInsertionCases(block, case105Def, case105Def.maxLookaheadCount);
    for (const auto &c : cases105)
      allCases.insert(c);

    // Case 106: XDL Write VGPR -> VM/VALU read/write VGPR (5 NOPs for 2-pass)
    auto case106Def = getCase106Definition();
    auto cases106 =
        findNopInsertionCases(block, case106Def, case106Def.maxLookaheadCount);
    for (const auto &c : cases106)
      allCases.insert(c);

    // TODO: cases 104, 107 - 120 (SGEMM, DGEMM variants)

    // Case ConservativeExtraDelays: Conservative extra delays after MFMA,
    // global_load, ds_read
    if (conservativeExtraDelays > 0) {
      auto caseConservativeExtraDelaysDef =
          getCaseConservativeExtraDelaysDefinition(conservativeExtraDelays);
      auto casesConservativeExtraDelays = findNopInsertionCases(
          block, caseConservativeExtraDelaysDef,
          caseConservativeExtraDelaysDef.maxLookaheadCount);
      for (const auto &caseConservativeExtraDelays :
           casesConservativeExtraDelays) {
        allCases.insert(caseConservativeExtraDelays);
      }
    }
  });

  if (allCases.empty())
    return;

  MLIRContext *ctx = allCases[0].inst1->getContext();
  OpBuilder builder(ctx);
  // Insert NOPs for all identified cases. Subtract existing instructions
  // between inst1 and inst2 from required NOPs, since those already count as
  // wait states. For VNOP cases, only VALU instructions count (ISA manual:
  // "NOPs or independent VALU instructions"). For SNOP cases, all instructions
  // count.
  for (const NopInsertionCase &nopCase : allCases) {
    int existingWaits;
    if (nopCase.nopType == NopType::VNOP) {
      // For VALU hazards, only VALU instructions count as wait states.
      // valuCount does NOT include inst2 itself (it's counted before the
      // selectInst2 check).
      existingWaits = nopCase.valuCount;
    } else {
      // For scalar hazards, all instructions count.
      existingWaits = std::max(0, nopCase.lookaheadCount - 1);
    }
    int actualNops = std::max(0, nopCase.requiredNops - existingWaits);
    if (actualNops == 0)
      continue;

    builder.setInsertionPoint(nopCase.inst2);

    if (nopCase.nopType == NopType::SNOP || nopCase.nopType == NopType::BOTH) {
      // Insert SNOPs. For cases requiring more than 15 NOPs, we insert multiple
      // SNOP instructions (max 15 wait states per SNOP).
      int remainingNops = actualNops;
      while (remainingNops > 0) {
        int nopValue = std::min(remainingNops, 15);
        inst::SOPPOp::create(builder, nopCase.inst2->getLoc(), OpCode::S_NOP,
                             static_cast<uint16_t>(nopValue));
        remainingNops -= nopValue;
      }
    }
    if (nopCase.nopType == NopType::VNOP || nopCase.nopType == NopType::BOTH) {
      int remainingNops = actualNops;
      // Insert VNOPs. VNOP doesn't support immediate values, so we insert one
      // VNOP per required NOP.
      for (int i = 0; i < remainingNops; ++i) {
        inst::VOP1NopOp::create(builder, nopCase.inst2->getLoc(),
                                OpCode::V_NOP);
      }
    }
  }
}
