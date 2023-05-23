// RUN: iree-compile %s --compile-to=vm --iree-execution-model=host-only | openxla-runner - token.main | FileCheck %s

module {
  func.func @await_token(%arg0: !async.token) {
    call @async.token.await(%arg0) : (!async.token) -> ()
    return
  }
  func.func private @async.token.await(!async.token)
}
