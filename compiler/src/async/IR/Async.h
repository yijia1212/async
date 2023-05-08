//===- Async.h - MLIR Async dialect -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the async dialect that is used for modeling asynchronous
// execution.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_SAMPLES_ASYNC_ASYNC_PLUGIN_IR_ASYNC_H_
#define IREE_SAMPLES_ASYNC_ASYNC_PLUGIN_IR_ASYNC_H_

#include "mlir/IR/Types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// Async Dialect
//===----------------------------------------------------------------------===//

#include "async/IR/AsyncOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Async Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "async/IR/AsyncOps.h.inc"


//===----------------------------------------------------------------------===//
// Async Dialect Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "async/IR/AsyncOpsTypes.h.inc"

#endif // IREE_SAMPLES_ASYNC_ASYNC_PLUGIN_IR_ASYNC_H_