// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

// HACK: this pokes in to private APIs for IO helpers while we expect
// applications to bring their own IO.
#include "iree/base/internal/file_io.h"

// Custom native module used in the sample.
// Modules may be linked in from native code or other bytecode modules loaded at
// runtime: there's no difference.
#include "openxla/runtime/async/loop_async.h"
#include "openxla/runtime/async/module.h"

void FreeLoop(iree_allocator_t allocator, iree_loop_t loop);

iree_status_t async_callback(void* user_data, iree_loop_t loop,
                             iree_status_t status, iree_vm_list_t* outputs) {
  if (iree_status_is_ok(status)) {
    fprintf(stdout, "Token Available\n");
    fflush(stdout);
  }

  iree_vm_list_release(outputs);
  return iree_ok_status();
}

// NOTE: CHECKs are dangerous but this is a sample; a real application would
// want to handle errors gracefully. We know in this constrained case that
// these won't fail unless something is catastrophically wrong (out of memory,
// solar flares, etc).
int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr,
            "Usage:\n"
            "  openxla-runner - <entry.point> # read from stdin\n"
            "  openxla-runner </path/to/say_hello.vmfb> <entry.point>\n");
    fprintf(stderr, "  (See the README for this sample for details)\n ");
    return -1;
  }

  // Internally IREE does not (in general) use malloc and instead uses the
  // provided allocator to allocate and free memory. Applications can integrate
  // their own allocator as-needed.
  iree_allocator_t allocator = iree_allocator_system();

  // Create the root isolated VM instance that we can create contexts within.
  iree_vm_instance_t* instance = NULL;
  IREE_CHECK_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                        allocator, &instance));
  IREE_CHECK_OK(openxla_async_runtime_module_register_types(instance));
  // Create the custom module that can be reused across contexts.
  iree_vm_module_t* async_runtime_module = NULL;
  IREE_CHECK_OK(iree_async_runtime_module_create(instance, allocator,
                                                 &async_runtime_module));

  // Load the module from stdin or a file on disk.
  // Applications can ship and load modules however they want (such as mapping
  // them into memory instead of allocating like this). Modules can also be
  // embedded in the binary but in those cases it makes more sense to use emitc
  // to avoid the bytecode entirely and have a fully static build (see
  // samples/emitc_modules/ for some examples).
  const char* module_path = argv[1];
  iree_file_contents_t* module_contents = NULL;
  if (strcmp(module_path, "-") == 0) {
    IREE_CHECK_OK(iree_stdin_read_contents(allocator, &module_contents));
  } else {
    IREE_CHECK_OK(
        iree_file_read_contents(module_path, allocator, &module_contents));
  }

  // Load the bytecode module from the vmfb.
  // This module can be reused across multiple contexts.
  // Note that we let the module retain the file contents for as long as needed.
  iree_vm_module_t* bytecode_module = NULL;
  IREE_CHECK_OK(iree_vm_bytecode_module_create(
      instance, module_contents->const_buffer,
      iree_file_contents_deallocator(module_contents), allocator,
      &bytecode_module));

  // Create the context for this invocation reusing the loaded modules.
  // Contexts hold isolated state and can be reused for multiple calls.
  // Note that the module order matters: the input user module is dependent on
  // the custom module.
  iree_vm_module_t* modules[] = {async_runtime_module, bytecode_module};
  iree_vm_context_t* context = NULL;
  IREE_CHECK_OK(iree_vm_context_create_with_modules(
      instance, IREE_VM_CONTEXT_FLAG_NONE, IREE_ARRAYSIZE(modules), modules,
      allocator, &context));

  // Lookup the function by fully-qualified name (module.func).
  iree_vm_function_t function;
  IREE_CHECK_OK(iree_vm_context_resolve_function(
      context, iree_make_cstring_view(argv[2]), &function));

  fprintf(stdout, "INVOKE BEGIN %s\n", argv[2]);
  fflush(stdout);
  // iree_vm_list_t* inputs = NULL;

  iree_vm_async_invoke_state_t state = {0};
  // // Setup call inputs with our buffers.
  // iree_status_t status = iree_vm_list_create(
  //     /*element_type=*/iree_vm_make_undefined_type_def(),
  //     /*capacity=*/1, allocator, &inputs);

  // if (!iree_status_is_ok(status)) {
  //   fprintf(stdout, "can't allocate input vm list");
  //   fflush(stdout);
  //   return 0;
  // }

  // iree_async_value_t* token = NULL;
  // iree_async_value_create_token(&token);
  // iree_vm_ref_t input_ref = iree_async_value_move_ref(token);
  // IREE_CHECK_OK(iree_vm_list_push_ref_move(inputs, &input_ref));

  iree_status_t loop_status = iree_ok_status();
  iree_loop_t loop = iree_loop_async(&loop_status);

  // status =
  iree_vm_async_invoke(loop, &state, context, function,
                       IREE_VM_INVOCATION_FLAG_NONE, /*policy=*/NULL, NULL,
                       /*outputs=*/NULL, allocator, async_callback, &state);

  // iree_async_value_signal(token);

  fprintf(stdout, "INVOKE END\n");
  fflush(stdout);

  // iree_vm_list_release(inputs);
  // inputs = NULL;
  iree_vm_context_release(context);
  iree_vm_module_release(bytecode_module);
  iree_vm_module_release(async_runtime_module);
  iree_vm_instance_release(instance);
  return 0;
}
