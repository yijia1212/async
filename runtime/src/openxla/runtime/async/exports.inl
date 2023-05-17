// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//         ██     ██  █████  ██████  ███    ██ ██ ███    ██  ██████
//         ██     ██ ██   ██ ██   ██ ████   ██ ██ ████   ██ ██
//         ██  █  ██ ███████ ██████  ██ ██  ██ ██ ██ ██  ██ ██   ███
//         ██ ███ ██ ██   ██ ██   ██ ██  ██ ██ ██ ██  ██ ██ ██    ██
//          ███ ███  ██   ██ ██   ██ ██   ████ ██ ██   ████  ██████
//
//===----------------------------------------------------------------------===//
//
// This file will be auto generated from async_runtime.imports.mlir in the future; for
// now it's modified by hand but with strict alphabetical sorting required.
// The order of these functions must be sorted ascending by name in a way
// compatible with iree_string_view_compare.
//
// Users are meant to `#define EXPORT_FN` to be able to access the information.
// #define EXPORT_FN(name, arg_type, ret_type, target_fn)

// clang-format off

EXPORT_FN("token.await", iree_async_runtime_module_token_await, r, v)
EXPORT_FN("token.create", iree_async_runtime_module_token_create, v, r)
EXPORT_FN("token.fail", iree_async_runtime_module_token_fail, r, v)
EXPORT_FN("token.query", iree_async_runtime_module_token_query, r, i)
EXPORT_FN("token.signal", iree_async_runtime_module_token_signal, r, v)

// EXPORT_FN("value.await", iree_async_runtime_module_value_await, r, v)
// EXPORT_FN("value.create", iree_async_runtime_module_value_create, II, r)
// EXPORT_FN("value.fail", iree_async_runtime_module_value_fail, r, v)
// EXPORT_FN("value.query", iree_async_runtime_module_value_query, r, i)
// EXPORT_FN("value.signal", iree_async_runtime_module_value_signal, r, v)

// clang-format on