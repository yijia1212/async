// RUN: iree-opt --iree-plugin=openxla_async --iree-print-plugin-info --pass-pipeline='builtin.module(openxla_async)' %s | FileCheck %s

// CHECK-LABEL: @await_token
func.func @await_token(%arg0: !async.token){
  // CHECK: async.token.await %arg0
  async.await %arg0 : !async.token
  return
}

// CHECK-LABEL: @await_value
func.func @await_value(%arg0: !async.value<i32>) -> i32 {
  // CHECK: async.value.await.i32 %arg0
  %0 = async.await %arg0 : !async.value<i32>
  return %0 : i32
}
