// RUN: iree-compile %s --iree-execution-model=host-only | openxla-runner - value.main | FileCheck %s

module attributes {vm.toplevel} {
  vm.module public @value {
    vm.import private @async.token.create() -> !vm.ref<!async.token>
    vm.import private @async.token.query(%token : !vm.ref<!async.token>) -> i32
    vm.import private @async.token.signal(%token : !vm.ref<!async.token>)
    vm.import private @async.token.fail(%token : !vm.ref<!async.token>, %status : i32)
    vm.import private @async.token.await(%token : !vm.ref<!async.token>)
    vm.import private @async.value.await.i32(%value : !vm.ref<!async.value<i32>>) -> i32
    vm.import private @async.value.await.i64(%value : !vm.ref<!async.value<i64>>) -> i64
    vm.import private @async.value.await.f32(%value : !vm.ref<!async.value<f32>>) -> f32
    vm.import private @async.value.await.f64(%value : !vm.ref<!async.value<f64>>) -> f64
    vm.func private @main(%arg0: !vm.ref<!async.value<i32>>) -> i32 {
      %0 = vm.call @async.value.await.i32(%arg0) : (!vm.ref<!async.value<i32>>) -> i32
      vm.return %0 : i32
    }
    vm.export @await_value
  }
}
