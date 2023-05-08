# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(ABSEIL_CPP_ROOT "${ASYNC_ROOT_DIR}/third_party/abseil-cpp/")

external_cc_library(
  PACKAGE
    cudnn_frontend
  NAME
    cudnn_frontend
  ROOT
    ${ABSEIL_CPP_ROOT}
  INCLUDES
    "${ABSEIL_CPP_ROOT}/absl"    
  HDRS
    "include/cudnn_frontend.h"
  PUBLIC
)