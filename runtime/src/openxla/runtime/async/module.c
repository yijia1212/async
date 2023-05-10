// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/runtime/async/module.h"
#include "openxla/runtime/async/api.h"

#include "iree/base/api.h"
#include "iree/vm/api.h"

#define IREE_ASYNC_RUNTIME_MODULE_VERSION_0_0 0x00000000u
#define IREE_ASYNC_RUNTIME_MODULE_VERSION_LATEST IREE_ASYNC_RUNTIME_MODULE_VERSION_0_0

//===----------------------------------------------------------------------===//
// Module type definitions
//===----------------------------------------------------------------------===//

typedef struct iree_async_runtime_module_t
{
  iree_allocator_t host_allocator;
} iree_async_runtime_module_t;

#define IREE_ASYNC_RUNTIME_MODULE_CAST(module) \
  (iree_async_runtime_module_t *)((uint8_t *)(module) + iree_vm_native_module_size());

typedef struct iree_async_runtime_module_state_t
{
  iree_allocator_t host_allocator;

} iree_async_runtime_module_state_t;

static void IREE_API_PTR iree_async_runtime_module_destroy(void *base_module)
{
}

static iree_status_t IREE_API_PTR
iree_async_runtime_module_alloc_state(void *self, iree_allocator_t host_allocator, iree_vm_module_state_t **out_module_state)
{
  iree_async_runtime_module_state_t *state = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*state), (void **)&state));
  memset(state, 0, sizeof(*state));
  state->host_allocator = host_allocator;

  *out_module_state = (iree_vm_module_state_t *)state;
  return iree_ok_status();
}

static void IREE_API_PTR
iree_async_runtime_module_free_state(void *self, iree_vm_module_state_t *module_state)
{
  iree_async_runtime_module_state_t *state = (iree_async_runtime_module_state_t *)module_state;
  iree_allocator_free(state->host_allocator, state);
}

//===----------------------------------------------------------------------===//
// iree_async_token_t
//===----------------------------------------------------------------------===//
IREE_VM_ABI_EXPORT(iree_async_runtime_module_token_create, //
                   iree_async_runtime_module_state_t,      //
                   v, r)
{
  iree_async_token_t *token = NULL;
  iree_status_t status =
      iree_async_token_create(&token);
  if (iree_status_is_ok(status))
  {
    rets->r0 = iree_async_token_move_ref(token);
  }
  else
  {
    iree_async_token_release(token);
  }

  return status;
}

IREE_VM_ABI_EXPORT(iree_async_runtime_module_token_query, //
                   iree_async_runtime_module_state_t,     //
                   r, i)
{
  iree_async_token_t *token = NULL;
  IREE_RETURN_IF_ERROR(iree_async_token_check_deref(args->r0, &token));

  iree_status_t query_status = iree_async_token_query(token);
  rets->i0 = iree_status_consume_code(query_status);
  iree_status_ignore(query_status);

  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_async_runtime_module_token_signal, //
                   iree_async_runtime_module_state_t,
                   r, v)
{
  iree_async_token_t *token = NULL;
  IREE_RETURN_IF_ERROR(iree_async_token_check_deref(args->r0, &token));
  return iree_async_token_signal(token);
}

IREE_VM_ABI_EXPORT(iree_async_runtime_module_token_fail, //
                   iree_async_runtime_module_state_t,
                   r, v)
{
  iree_async_token_t *token = NULL;
  IREE_RETURN_IF_ERROR(iree_async_token_check_deref(args->r0, &token));
  iree_async_token_fail(token);
  return iree_ok_status();
}

// Enters a wait frame for all async tokens in |tokens|.
// Returns an |out_wait_status| of OK if all tokens are available or
// IREE_STATUS_DEFERRED if one or more tokens are still pending and a wait
// frame was entered.
static iree_status_t iree_async_runtime_module_token_await_begin(
    iree_vm_stack_t *stack, iree_async_token_t *token, iree_status_t *out_wait_status)
{
  iree_vm_wait_frame_t *wait_frame = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_stack_wait_enter(stack, IREE_VM_WAIT_ALL, 1, iree_infinite_timeout(), 0, &wait_frame));

  iree_wait_source_t wait_source = iree_async_token_await(token);
  wait_frame->wait_sources[0] = wait_source;
  wait_frame->count = 1;

  *out_wait_status = iree_status_from_code(IREE_STATUS_DEFERRED);
  return iree_ok_status();
}

enum iree_async_runtime_module_token_await_pc_e
{
  IREE_ASYNC_RUNTIME_MODULE_TOKEN_AWAIT_PC_BEGIN = 0,
  IREE_ASYNC_RUNTIME_MODULE_TOKEN_AWAIT_PC_RESUME,
};

IREE_VM_ABI_EXPORT(iree_async_runtime_module_token_await, //
                   iree_async_runtime_module_state_t,     //
                   r, v)
{
  iree_vm_stack_frame_t *current_frame = iree_vm_stack_top(stack);
  iree_status_t wait_status = iree_ok_status();

  if (current_frame->pc == IREE_ASYNC_RUNTIME_MODULE_TOKEN_AWAIT_PC_BEGIN)
  {
    iree_async_token_t *token = NULL;
    IREE_RETURN_IF_ERROR(iree_async_token_check_deref(args->r0, &token));
    current_frame->pc = IREE_ASYNC_RUNTIME_MODULE_TOKEN_AWAIT_PC_RESUME;
    IREE_RETURN_IF_ERROR(iree_async_runtime_module_token_await_begin(stack, token, &wait_status));
  }
  else
  {
    // Resume by leaving the wait frame and storing the result.
    iree_vm_wait_result_t wait_result;
    IREE_RETURN_IF_ERROR(iree_vm_stack_wait_leave(stack, &wait_result));
    wait_status = wait_result.status;
  }

  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(wait_status))
  {
    // Successful wait.
    // rets->i0 = 0;
  }
  else if (iree_status_is_deferred(wait_status))
  {
    // Yielding; resume required.
    // NOTE: zone not ended as it's reserved on the stack.
    status = wait_status;
  }
  else
  {
    // Fail the invocation
    status = wait_status;
  }

  return status;
}

//===----------------------------------------------------------------------===//
// VM module interface implementation
//===----------------------------------------------------------------------===//

// NOTE: this must match the ordering of the iree_hal_module_exports_ table.
static const iree_vm_native_function_ptr_t iree_async_runtime_module_funcs_[] = {
#define EXPORT_FN(name, target_fn, arg_types, ret_types)       \
  {                                                            \
      .shim = (iree_vm_native_function_shim_t)                 \
          iree_vm_shim_##arg_types##_##ret_types,              \
      .target = (iree_vm_native_function_target_t)(target_fn), \
  },
#include "exports.inl" // IWYU pragma: keep
#undef EXPORT_FN
};

static const iree_vm_native_export_descriptor_t iree_async_runtime_module_exports_[] = {
#define EXPORT_FN(name, target_fn, arg_types, ret_types)           \
  {                                                                \
      .local_name = iree_string_view_literal(name),                \
      .calling_convention =                                        \
          iree_string_view_literal("0" #arg_types "_" #ret_types), \
      .attr_count = 0,                                             \
      .attrs = NULL,                                               \
  },
#include "exports.inl" // IWYU pragma: keep
#undef EXPORT_FN
};

static_assert(IREE_ARRAYSIZE(iree_async_runtime_module_funcs_) ==
                  IREE_ARRAYSIZE(iree_async_runtime_module_exports_),
              "function pointer table must be 1:1 with exports");

// NOTE: 0 length, but can't express that in C.
static const iree_vm_native_import_descriptor_t iree_async_runtime_module_imports_[1];

static const iree_vm_native_module_descriptor_t iree_async_runtime_module_descriptor_ = {
    .name = iree_string_view_literal("async"),
    .version = IREE_ASYNC_RUNTIME_MODULE_VERSION_LATEST,
    .attr_count = 0,
    .attrs = NULL,
    .dependency_count = 0,
    .dependencies = NULL,
    .import_count = 0, // workaround for 0-length C struct
    .imports = iree_async_runtime_module_imports_,
    .export_count = IREE_ARRAYSIZE(iree_async_runtime_module_exports_),
    .exports = iree_async_runtime_module_exports_,
    .function_count = IREE_ARRAYSIZE(iree_async_runtime_module_funcs_),
    .functions = iree_async_runtime_module_funcs_,
};

IREE_API_EXPORT iree_status_t iree_async_runtime_module_create(
    iree_vm_instance_t *instance, iree_allocator_t host_allocator,
    iree_vm_module_t **out_module)
{
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;

  // Setup the interface with the functions we implement ourselves. Any function
  // we omit will be handled by the base native module.
  static const iree_vm_module_t interface = {
      .destroy = iree_async_runtime_module_destroy,
      .alloc_state = iree_async_runtime_module_alloc_state,
      .free_state = iree_async_runtime_module_free_state,
  };

  // Allocate shared module state.
  iree_host_size_t total_size =
      iree_vm_native_module_size() + sizeof(iree_async_runtime_module_t);
  iree_vm_module_t *base_module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void **)&base_module));
  memset(base_module, 0, total_size);
  iree_status_t status =
      iree_vm_native_module_initialize(&interface, &iree_async_runtime_module_descriptor_,
                                       instance, host_allocator, base_module);
  if (!iree_status_is_ok(status))
  {
    iree_allocator_free(host_allocator, base_module);
    return status;
  }

  iree_async_runtime_module_t *module = IREE_ASYNC_RUNTIME_MODULE_CAST(base_module);
  module->host_allocator = host_allocator;

  *out_module = base_module;
  return iree_ok_status();
}
