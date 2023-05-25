// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_RUNTIME_ASYNC_ASYNCRUNTIME_CC_H_
#define OPENXLA_RUNTIME_ASYNC_ASYNCRUNTIME_CC_H_

#include "iree/vm/ref_cc.h"
#include "openxla/runtime/async/async_runtime_util.h"
#include "tfrt/concurrency/async_value.h"
#include "tfrt/concurrency/async_value_ref.h"
#include "tfrt/concurrency/chain.h"

namespace openxla::runtime::async {

class AsyncValue : public iree::vm::RefObject<AsyncValue> {
 public:
  explicit AsyncValue(tsl::AsyncValueRef<tsl::Chain> &&u)
      : type_(IREE_VM_VALUE_TYPE_NONE), value_(u.ReleaseRCRef()) {}

  template <typename T, EnableIfScalarType<T> * = nullptr>
  explicit AsyncValue(tsl::AsyncValueRef<T> &&u)
      : type_(NativeToVMValueType<T>()), value_(u.ReleaseRCRef()) {}

  template <typename T>
  const T &get() {
    return value_->get<T>();
  }

  tsl::AsyncValue *GetAsyncValue() { return value_.get(); }

  bool IsError() const { return value_->IsError(); }

 private:
  iree_vm_value_type_t type_;
  tsl::RCReference<tsl::AsyncValue> value_;
};

template <typename T>
static AsyncValue *AsValue(tsl::AsyncValueRef<T> value) {
  return new AsyncValue(std::move(value));
}

}  // namespace openxla::runtime::async

#endif  // OPENXLA_RUNTIME_ASYNC_ASYNCRUNTIME_CC_H_