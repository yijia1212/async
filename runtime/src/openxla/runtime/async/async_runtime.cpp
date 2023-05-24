// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/runtime/async/async_runtime.h"

#include <cstddef>
#include <optional>

#include "iree/vm/ref_cc.h"
#include "iree/vm/value.h"
#include "openxla/runtime/async/async_runtime_util.h"
#include "tfrt/concurrency/async_value.h"
#include "tfrt/concurrency/async_value_ref.h"
#include "tfrt/concurrency/chain.h"

namespace openxla::runtime::async {

class AsyncValue : public iree::vm::RefObject<AsyncValue> {
 public:
  explicit AsyncValue(tsl::AsyncValueRef<tsl::Chain> value)
      : type_(IREE_VM_VALUE_TYPE_NONE), val_(value.GetAsyncValue()) {}

  template <typename T, EnableIfScalarType<T> * = nullptr>
  explicit AsyncValue(tsl::AsyncValueRef<T> value)
      : type_(NativeToVMValueType<T>()), val_(value.GetAsyncValue()) {}

  template <typename T>
  const T &get() {
    return val_->get<T>();
  }

  tsl::AsyncValue *GetAsyncValue() { return val_; }

  bool IsError() const { return val_->IsError(); }

 private:
  iree_vm_value_type_t type_;
  tsl::AsyncValue *val_;
};
}  // namespace openxla::runtime::async

using openxla::runtime::async::AsyncValue;

IREE_API_EXPORT iree_status_t
iree_async_value_create_token(iree_async_value_t **out_token) {
  tsl::AsyncValueRef<tsl::Chain> chain =
      tsl::MakeConstructedAsyncValueRef<tsl::Chain>();
  AsyncToken *val = new AsyncToken(chain);
  *out_token = reinterpret_cast<iree_async_token_t *>(val);
  return iree_ok_status();
}

IREE_API_EXPORT void iree_async_value_destroy(iree_async_value_t *value) {
  AsyncValue *val = reinterpret_cast<AsyncValue *>(value);
  IREE_ASSERT_ARGUMENT(val);
  delete val;
}

IREE_API_EXPORT uint32_t iree_async_value_offsetof_counter() {
  return AsyncValue::offsetof_counter();
}

IREE_API_EXPORT void iree_async_value_release(iree_async_value_t *value) {
  AsyncValue *val = reinterpret_cast<AsyncValue *>(value);
  val->ReleaseReference();
}

IREE_API_EXPORT iree_status_t
iree_async_token_query(iree_async_token_t *token) {
  AsyncToken *val = reinterpret_cast<AsyncToken *>(token);
  if (!val) return iree_ok_status();
  if (!val->GetAsyncValue()->IsAvailable()) {
    return iree_status_from_code(IREE_STATUS_DEFERRED);
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_async_token_signal(iree_async_token_t *token) {
  AsyncToken *val = reinterpret_cast<AsyncToken *>(token);
  val->GetAsyncValue()->SetStateConcrete();
  return iree_ok_status();
}

IREE_API_EXPORT void iree_async_token_fail(iree_async_token_t *token) {
  AsyncToken *val = reinterpret_cast<AsyncToken *>(token);
  val->GetAsyncValue()->SetError(absl::InternalError("async runtime error"));
}

IREE_API_EXPORT iree_status_t iree_async_token_wait(iree_async_token_t *token,
                                                    iree_timeout_t timeout) {
  assert(false && "unimplemented");
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_async_token_and_then(iree_async_token_t *token,
                          iree_loop_callback_t callback, iree_loop_t loop) {
  AsyncToken *val = reinterpret_cast<AsyncToken *>(token);
  val->GetAsyncValue()->AndThen([callback, loop]() {
    iree_status_t status =
        callback.fn(callback.user_data, loop, iree_ok_status());
    (void)status;
    // notify loop is status is not OK
  });
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_async_token_wait_source_ctl(
    iree_wait_source_t wait_source, iree_wait_source_command_t command,
    const void *params, void **inout_ptr) {
  iree_async_token_t *token = (iree_async_token_t *)wait_source.self;
  switch (command) {
    case IREE_WAIT_SOURCE_COMMAND_QUERY: {
      iree_status_code_t *out_wait_status_code =
          (iree_status_code_t *)inout_ptr;
      iree_status_t status = iree_async_token_query(token);
      if (!iree_status_is_ok(status)) {
        *out_wait_status_code = iree_status_code(status);
        iree_status_ignore(status);
      } else {
        *out_wait_status_code = IREE_STATUS_OK;
      }
      return iree_ok_status();
    }
    case IREE_WAIT_SOURCE_COMMAND_WAIT_ONE: {
      const iree_timeout_t timeout =
          ((const iree_wait_source_wait_params_t *)params)->timeout;
      return iree_async_token_wait(token, timeout);
    }
    case IREE_WAIT_SOURCE_COMMAND_EXPORT: {
      const iree_wait_primitive_type_t target_type =
          ((const iree_wait_source_export_params_t *)params)->target_type;
      // TODO(benvanik): support exporting fences to real wait handles.
      iree_wait_primitive_t *out_wait_primitive =
          (iree_wait_primitive_t *)inout_ptr;
      memset(out_wait_primitive, 0, sizeof(*out_wait_primitive));
      (void)target_type;
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "requested wait primitive type %d is unavailable",
                              (int)target_type);
    }
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unimplemented wait_source command");
  }
}

IREE_API_EXPORT iree_wait_source_t
iree_async_token_await(iree_async_token_t *token) {
  if (!token) return iree_wait_source_immediate();
  iree_wait_source_t wait_source;
  wait_source.self = token;
  wait_source.data = 0;
  wait_source.ctl = iree_async_token_wait_source_ctl;
  return wait_source;
}

IREE_API_EXPORT iree_status_t iree_async_value_get_available_value(
    iree_async_value_t *value, iree_host_size_t buffer_capacity, char *buffer) {
  assert(false && "unimplemented");
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_async_value_create(iree_async_value_t **out_value) {
  assert(false && "unimplemented");
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_async_value_query(iree_async_value_t *value) {
  assert(false && "unimplemented");
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_async_value_signal(iree_async_value_t *value) {
  assert(false && "unimplemented");
  return iree_ok_status();
}

IREE_API_EXPORT void iree_async_value_fail(iree_async_value_t *value) {
  assert(false && "unimplemented");
}

IREE_API_EXPORT iree_status_t iree_async_value_wait(iree_async_value_t *value,
                                                    iree_timeout_t timeout) {
  assert(false && "unimplemented");
  return iree_ok_status();
}

// Releases |token| and destroys it if the caller is the last owner.
IREE_API_EXPORT void iree_async_value_release(iree_async_value_t *value) {
  assert(false && "unimplemented");
}

IREE_API_EXPORT iree_status_t
iree_async_value_and_then(iree_async_value_t *value,
                          iree_loop_callback_t callback, iree_loop_t loop) {
  assert(false && "unimplemented");
  return iree_ok_status();
}

IREE_API_EXPORT iree_wait_source_t
iree_async_value_await(iree_async_value_t *value) {
  // if (!value) return iree_wait_source_immediate();
  // return (iree_wait_source_t){
  //     .self = value,
  //     .data = 0,
  //     .ctl = iree_async_value_wait_source_ctl,
  // };
  return iree_wait_source_immediate();
}

IREE_API_EXPORT iree_status_t iree_async_value_wait_source_ctl(
    iree_wait_source_t wait_source, iree_wait_source_command_t command,
    const void *params, void **inout_ptr) {
  assert(false && "unimplemented");
  return iree_ok_status();
}

IREE_API_EXPORT void iree_async_value_destroy(iree_async_value_t *value) {
  assert(false && "unimplemented");
}
IREE_API_EXPORT uint32_t iree_async_value_offsetof_counter() {
  assert(false && "unimplemented");
  return 0;
}