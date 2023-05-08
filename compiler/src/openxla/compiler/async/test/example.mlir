// RUN: iree-opt --iree-plugin=async --iree-print-plugin-info --pass-pipeline='builtin.module(iree-simpleio-legalize)' %s | FileCheck %s

// CHECK-LABEL: @await_token
func.func @await_token(%arg0: !async.token) ->!async.token{
  // CHECK: async.await %arg0
  async.await %arg0 : !async.token
  return
}

//func.func @await_token(%arg0: !async.token) -> !async.token {
//^entry:
//  %0 = call @async.runtime.create_token : !async.token
//  %1 = call @async.runtime.await_and_resume(%arg0) : (!async.token) -> i1
//  br %1 ^yield, ^continue
//^yield:
//  vm.yield
//  vm.br ^continue
//^continue:
//  %2 = call @async.runtime.is_error %arg0 : !async.token
//  vm.cond_br %2, ^bb1, ^bb2
//^bb1:
//  call @async.runtime.set_error %0 : !async.token
//  vm.br ^bb3
//^bb2:
//  call @async.runtime.set_available %0 : !async.token
//  vm.br ^bb3 
//^bb3:
//  return %0 : !async.token
//}