# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_cc_library(
  NAME
    defs
  INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_BINARY_DIR}/..
  PUBLIC
)

iree_cc_library(
  NAME
    async_value
  HDRS
    "async_value.h"
    "async_value_ref.h"
    "chain.h"
  SRCS
    "async_value.cc"
    "async_value_ref.cc"
  DEPS
    ::defs
    ::concurrent_vector
    ::ref_count
    absl::base
    absl::status
  PUBLIC
)

iree_cc_library(
  NAME
    concurrent_vector
  HDRS
    "concurrent_vector.h"
  DEPS
    ::defs
    absl::base
    absl::status
  PUBLIC
)

iree_cc_library(
  NAME
    ref_count
  HDRS
    "ref_count.h"
  DEPS
    ::defs
  PUBLIC
)
