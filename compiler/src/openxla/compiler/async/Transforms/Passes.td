//===-- Passes.td - Async pass definition file -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPENXLA_ASYNC_TRANSFORMS_PASSES_TD_
#define OPENXLA_ASYNC_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def AsyncToAsyncRuntime : Pass<"async-to-async-runtime", "mlir::ModuleOp"> {
  let summary = "Lower all high level async operations to"
                "the explicit async.runtime operations";
  let constructor = "openxla::compiler::async::createConvertAsyncOpsPass()";
  let dependentDialects = ["::openxla::compiler::async::AsyncDialect", "func::FuncDialect"];
}

#endif // OPENXLA_ASYNC_TRANSFORMS_PASSES_TD_