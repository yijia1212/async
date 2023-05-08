// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ASYNC_DIALECT_TD
#define ASYNC_DIALECT_TD

include "mlir/IR/OpBase.td"

def AsyncDialect : Dialect {
  let name = "async";
  let cppNamespace = "::mlir::iree_compiler::IREE::Async";

  let summary = "Types and operations for async dialect";
  let description = [{
    This dialect contains operations for modeling asynchronous execution.
  }];

  let useDefaultTypePrinterParser = 1;
}

#endif // ASYNC_DIALECT_TD