// RUN: iree-compile %s --iree-execution-model=host-only | openxla-runner - value.main | FileCheck %s

module {
  func.func @await_scalar_value(%arg0: !async.value) -> i32 {
    %0 = call @async.value.await.i32(%arg0) : (!async.value) -> i32
    return %0 : i32
  }
  func.func private @async.value.await.i32(!async.value) -> i32
}
