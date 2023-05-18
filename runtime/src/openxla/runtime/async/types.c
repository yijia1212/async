// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/runtime/async/types.h"

//===----------------------------------------------------------------------===//
// Type wrappers
//===----------------------------------------------------------------------===//

IREE_VM_DEFINE_TYPE_ADAPTERS(iree_async_token, iree_async_token_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_async_value, iree_async_value_t);

//===----------------------------------------------------------------------===//
// Type registration
//===----------------------------------------------------------------------===//
#define IREE_VM_REGISTER_ASYNC_C_TYPE(instance, type, name, destroy_fn, \
                                      registration)                     \
  static const iree_vm_ref_type_descriptor_t registration##_storage = { \
      .type_name = IREE_SVL(name),                                      \
      .offsetof_counter = 0 / IREE_VM_REF_COUNTER_ALIGNMENT,            \
      .destroy = (iree_vm_ref_destroy_t)destroy_fn,                     \
  };                                                                    \
  IREE_RETURN_IF_ERROR(iree_vm_instance_register_type(                  \
      instance, &registration##_storage, &registration));

IREE_API_EXPORT iree_status_t
iree_async_runtime_module_register_all_types(iree_vm_instance_t* instance) {
  IREE_VM_REGISTER_ASYNC_C_TYPE(instance, iree_async_token_t, "async.token",
                                iree_async_token_destroy,
                                iree_async_token_registration);
  IREE_VM_REGISTER_ASYNC_C_TYPE(instance, iree_async_value_t, "async.value",
                                iree_async_value_destroy,
                                iree_async_value_registration);
  return iree_ok_status();
}

#define IREE_VM_RESOLVE_ASYNC_C_TYPE(instance, type, name, registration)    \
  registration =                                                            \
      iree_vm_instance_lookup_type(instance, iree_make_cstring_view(name)); \
  if (!registration) {                                                      \
    return iree_make_status(IREE_STATUS_INTERNAL,                           \
                            "VM type `" name                                \
                            "` not registered with the instance");          \
  }

IREE_API_EXPORT iree_status_t
iree_async_runtime_module_resolve_all_types(iree_vm_instance_t* instance) {
  IREE_VM_RESOLVE_ASYNC_C_TYPE(instance, iree_async_token_t, "async.token",
                               iree_async_token_registration);
  IREE_VM_RESOLVE_ASYNC_C_TYPE(instance, iree_async_value_t, "async.value",
                               iree_async_value_registration);
  return iree_ok_status();
}