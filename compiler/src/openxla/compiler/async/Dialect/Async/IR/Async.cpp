//===- Async.cpp - MLIR Async Operations ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Async.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace openxla::compiler::Async;

#include "AsyncOpsDialect.cpp.inc"

void AsyncDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "AsyncOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "AsyncOpsTypes.cpp.inc"
      >(); 
}
