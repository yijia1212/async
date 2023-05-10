// RUN: iree-opt --iree-plugin=async --iree-print-plugin-info --pass-pipeline='builtin.module(iree-simpleio-legalize)' %s | FileCheck %s

// CHECK-LABEL: @await_token
func.func @await_token(%arg0: !async.token) ->!async.token{
  // CHECK: async.token.await %arg0
  async.await %arg0 : !async.token
  return
}

