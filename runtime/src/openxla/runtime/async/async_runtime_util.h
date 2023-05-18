// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_RUNTIME_ASYNC_ASYNCRUNTIME_UTIL_H_
#define OPENXLA_RUNTIME_ASYNC_ASYNCRUNTIME_UTIL_H_

#include <cstddef>

#include "concurrency/async_value.h"
#include "concurrency/async_value_ref.h"

namespace openxla::runtime::async {
struct AsyncToken;
struct AsyncValue;

template <typename T>
AsyncValue* AsValue(
    tsl::AsyncValueRef<T> value, size_t size, size_t alignment,
    absl::FunctionRef<void(const T*, std::byte* storage)> write) {
  Value* runtime_async_value = AsyncRuntime::CreateValue(size, alignment);
  value.AndThen([runtime_async_value, write](absl::StatusOr<T*> status_or) {
    if (!status_or.ok()) {
      AsyncRuntime::SetError(runtime_async_value);
    } else {
      auto* store = AsyncRuntime::GetStorage(runtime_async_value);
      write(*status_or, store);
      AsyncRuntime::SetAvailable(runtime_async_value);
    }
  });
  return runtime_async_value;
}

template <typename T>
AsyncValue* AsValue(
    tsl::AsyncValueRef<T> value,
    absl::FunctionRef<std::pair<size_t, size_t>(const T*)> size_and_alignment,
    absl::FunctionRef<void(const T*, std::byte* storage)> write) {
  AsyncValue* runtime_async_value = AsyncRuntime::CreateValue();
  value.AndThen([runtime_async_value, size_and_alignment,
                 write](absl::StatusOr<T*> status_or) {
    if (!status_or.ok()) {
      AsyncRuntime::SetError(runtime_async_value);
    } else {
      auto size_alignment = size_and_alignment(*status_or);
      AsyncRuntime::AllocateStorage(runtime_async_value, size_alignment.first,
                                    size_alignment.second);
      auto* store = AsyncRuntime::GetStorage(runtime_async_value);
      write(*status_or, store);
      AsyncRuntime::SetAvailable(runtime_async_value);
    }
  });
  return runtime_async_value;
}
}  // namespace openxla::runtime::async

#endif  // OPENXLA_RUNTIME_ASYNC_ASYNCRUNTIME_UTIL_H_