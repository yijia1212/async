//===-- Passes.td - Async pass definition file -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ASYNC_PASSES
#define MLIR_DIALECT_ASYNC_PASSES

include "mlir/Pass/PassBase.td"

def AsyncToAsyncRuntime : Pass<"async-to-async-runtime", "ModuleOp"> {
  let summary = "Lower all high level async operations (e.g. async.execute) to"
                "the explicit async.runtime and async.coro operations";
  let constructor = "mlir::createAsyncToAsyncRuntimePass()";
  let dependentDialects = ["async::AsyncDialect", "func::FuncDialect"];
}

#endif // MLIR_DIALECT_ASYNC_PASSES