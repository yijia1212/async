//===- Passes.h - Async pass entry points -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_SAMPLES_ASYNC_ASYNC_PLUGIN_TRANSFORMS_PASSES_H_
#define IREE_SAMPLES_ASYNC_ASYNC_PLUGIN_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;
class ConversionTarget;

#define GEN_PASS_DECL
#include "samples/async/async_plugin/src/Transforms/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createAsyncToAsyncRuntimePass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "samples/async/async_plugin/src/Transforms/Passes.h.inc"

} // namespace mlir

#endif // IREE_SAMPLES_ASYNC_ASYNC_PLUGIN_TRANSFORMS_PASSES_H_