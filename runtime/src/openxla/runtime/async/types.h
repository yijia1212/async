// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_RUNTIME_ASYNC_TYPES_H_
#define OPENXLA_RUNTIME_ASYNC_TYPES_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/vm/api.h"
#include "openxla/runtime/async/api.h"

IREE_VM_DECLARE_TYPE_ADAPTERS(iree_async_token, iree_async_token_t);
IREE_VM_DECLARE_TYPE_ADAPTERS(iree_async_value, iree_async_value_t);

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Registers the custom types used by the full async module.
// WARNING: not thread-safe; call at startup before using.
IREE_API_EXPORT iree_status_t
iree_async_runtime_module_register_all_types(iree_vm_instance_t* instance);

IREE_API_EXPORT iree_status_t
iree_async_runtime_module_resolve_all_types(iree_vm_instance_t* instance);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MODULES_HAL_TYPES_H_