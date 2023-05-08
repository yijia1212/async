//===- AsyncRuntime.h - Async runtime reference implementation ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares basic Async runtime API for supporting Async dialect
// to LLVM dialect lowering.
//
//===----------------------------------------------------------------------===//

#ifndef OPENXLA_RUNTIME_ASYNC_ASYNCRUNTIME_H_
#define OPENXLA_RUNTIME_ASYNC_ASYNCRUNTIME_H_

#include "iree/base/api.h"
#include "iree/base/status.h"

typedef struct iree_async_token_t iree_async_token_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

IREE_API_EXPORT iree_status_t iree_async_token_create(iree_async_token_t** out_token);

IREE_API_EXPORT iree_status_t iree_async_token_query(iree_async_token_t* token);

IREE_API_EXPORT iree_status_t iree_async_token_signal(iree_async_token_t* token);

IREE_API_EXPORT void iree_async_token_fail(iree_async_token_t* token);

IREE_API_EXPORT iree_status_t iree_async_token_wait(iree_async_token_t* token,
                                                  iree_timeout_t timeout);

// Releases |token| and destroys it if the caller is the last owner.
IREE_API_EXPORT void iree_async_token_release(iree_async_token_t* token);

IREE_API_EXPORT iree_status_t
iree_async_token_and_then(iree_async_token_t* token, iree_loop_callback_t callback, iree_loop_t loop);                                                  

// Returns a wait source reference to |async_token|
// The async_token must be kept live for as long as the reference is live
IREE_API_EXPORT iree_wait_source_t 
iree_async_token_await(iree_async_token_t* token);

IREE_API_EXPORT iree_status_t 
iree_async_token_wait_source_ctl(iree_wait_source_t wait_source,
  iree_wait_source_command_t command, const void* params, void** inout_ptr); 

//===----------------------------------------------------------------------===//
// iree_hal_fence_t implementation details
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_async_token_destroy(iree_async_token_t* token);
IREE_API_EXPORT uint32_t
iree_async_token_offsetof_counter();

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // OPENXLA_RUNTIME_ASYNC_ASYNCRUNTIME_H_