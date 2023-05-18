// RUN: iree-compile %s --compile-to=vm --iree-execution-model=host-only | openxla-runner - token.main | FileCheck %s

module attributes {vm.toplevel} {
  vm.module public @token {
    vm.import private @async.token.create() -> !vm.ref<!async.token>
    vm.import private @async.token.query(%token : !vm.ref<!async.token>) -> i32
    vm.import private @async.token.signal(%token : !vm.ref<!async.token>)
    vm.import private @async.token.fail(%token : !vm.ref<!async.token>, %status : i32)
    vm.import private @async.token.await(%token : !vm.ref<!async.token>)
    vm.func private @main(%arg0: !vm.ref<!async.token>) {
      vm.call @async.token.await(%arg0) : (!vm.ref<!async.token>) -> ()
      vm.return
    }
    vm.export @main
  }
}
