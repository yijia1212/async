# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_cc_library(
  NAME
    Transforms
  HDRS
    "Passes.h"
    "Passes.h.inc"
  SRCS
    "AsyncToAsyncRuntime.cpp"
  DEPS
    ::PassesIncGen
    ::async_runtime_imports
    MLIRIR
    MLIRPass
    iree::compiler::Dialect::VM::Conversion
    iree::compiler::Dialect::VM::IR
    openxla::compiler::async::Dialect::Async::IR
    openxla::compiler::async::defs
  PUBLIC
)

iree_tablegen_library(
  NAME
    PassesIncGen
  TD_FILE
    "Passes.td"
  OUTS
    --gen-pass-decls Passes.h.inc
)

iree_c_embed_data(
  NAME
    async_runtime_imports
  SRCS
    "async_runtime.imports.mlir"
  C_FILE_OUTPUT
    "async_runtime.imports.c"
  H_FILE_OUTPUT
    "async_runtime.imports.h"
  IDENTIFIER
    "iree_async_runtime_imports"
  FLATTEN
  PUBLIC
)