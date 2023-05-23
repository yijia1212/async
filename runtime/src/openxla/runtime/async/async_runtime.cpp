// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/runtime/async/async_runtime.h"

#include <cstddef>
#include <optional>

#include "iree/vm/ref_cc.h"
#include "tfrt/concurrency/async_value.h"
#include "tfrt/concurrency/async_value_ref.h"
#include "tfrt/concurrency/chain.h"

namespace openxla::runtime::async {

using tsl::AsyncValueOwningRef;
using tsl::Chain;
using tsl::MakeAvailableAsyncValueRef;
using tsl::MakeConstructedAsyncValueRef;
using tsl::internal::AsyncValueStorage;

struct AsyncToken : public iree::vm::RefObject<AsyncToken> {
  explicit AsyncToken() : chain(MakeConstructedAsyncValueRef<Chain>(storage)) {}

  tsl::AsyncValue *GetAsyncValue() const { return chain.AsPtr().value(); }

  AsyncValueStorage<Chain> storage;
  AsyncValueOwningRef<Chain> chain;
};

struct AsyncValue : public iree::vm::RefObject<AsyncValue> {
  explicit AsyncValue() : chain(MakeConstructedAsyncValueRef<Chain>(storage)) {}

  explicit AsyncValue(size_t size, size_t alignment)
      : data_storage(Storage(size, alignment)),
        chain(MakeConstructedAsyncValueRef<Chain>(storage)) {}

  std::byte *GetStorage() {
    assert(!GetAsyncValue()->IsError() && "unexpected error state");
    assert(data_storage.has_value() && "unallocated data storage");
    if (data_storage->is_inline) return &data_storage->inline_buffer[0];
    return data_storage->allocated_buffer;
  }

  void AllocateStorage(size_t size, size_t alignment) {
    data_storage = Storage(size, alignment);
    // Storage memory will be initialized by the compiled executable.
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(GetStorage(), size);
  }

  tsl::AsyncValue *GetAsyncValue() const { return chain.AsPtr().value(); }

  // If the requested async value storage is small, use the inlined storage.
  // Fall back on dynamic allocation if the requested storage size is large.
  struct Storage {
    static const int kSize = 128;  // enough to fit memref descriptor of rank 5
    static const int kAlign = alignof(std::max_align_t);

    Storage(size_t size, size_t alignment)
        : is_inline(CanStoreInline(size, alignment)) {
      if (!is_inline)
        // TODO: Handle AlignedMalloc
        allocated_buffer = reinterpret_cast<std::byte *>(malloc(size));
    }

    ~Storage() {
      // TODP: Handle AlignedFree
      if (!is_inline) free(allocated_buffer);
    }

    static bool CanStoreInline(size_t size, size_t alignment) {
      assert(absl::has_single_bit(alignment));
      return size <= kSize && alignment <= kAlign;
    }

    bool is_inline;
    union {
      alignas(kAlign) std::array<std::byte, kSize> inline_buffer;
      std::byte *allocated_buffer;
    };
  };

  std::optional<Storage> data_storage;

  // Async value that tracks value readiness. It becomes available when result
  // is written to the data storage and ready for consumption.
  AsyncValueStorage<Chain> storage;
  AsyncValueOwningRef<Chain> chain;
};

}  // namespace openxla::runtime::async

using openxla::runtime::async::AsyncToken;
using openxla::runtime::async::AsyncValue;

IREE_API_EXPORT iree_status_t
iree_async_token_create(iree_async_token_t **out_token) {
  AsyncToken *val = new AsyncToken();
  *out_token = reinterpret_cast<iree_async_token_t *>(val);
  return iree_ok_status();
}

IREE_API_EXPORT void iree_async_token_destroy(iree_async_token_t *token) {
  AsyncToken *val = reinterpret_cast<AsyncToken *>(token);
  IREE_ASSERT_ARGUMENT(val);
  delete val;
}

IREE_API_EXPORT uint32_t iree_async_token_offsetof_counter() {
  return AsyncToken::offsetof_counter();
}

IREE_API_EXPORT void iree_async_token_release(iree_async_token_t *token) {
  AsyncToken *val = reinterpret_cast<AsyncToken *>(token);
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
  // if (!token) return iree_wait_source_immediate();
  // return (iree_wait_source_t){
  //     .self = token,
  //     .data = 0,
  //     .ctl = iree_async_token_wait_source_ctl,
  // };

  return iree_wait_source_immediate();
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