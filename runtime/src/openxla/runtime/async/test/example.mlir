// RUN: iree-compile %s --iree-execution-model=host-only | openxla-runner - example.main | FileCheck %s

module @example {
  func.func private @async.token.await(!async.token)

  // CHECK-LABEL: INVOKE BEGIN example.main
  func.func @main(%arg0: !async.token) {
    call @async.token.await(%arg0) : (!async.token) -> ()
    return
  }
  // CHECK-NEXT: INVOKE END
}
