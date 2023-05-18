// IREE Async runtime module imports.
//
// This is embedded in the compiler binary and inserted into any module
// containing Async dialect ops (async.*) that is lowered to the VM dialect.
vm.module @async {
//===----------------------------------------------------------------------===//
// iree_async_token_t
//===----------------------------------------------------------------------===//

// Returns an unavailable token
vm.import private @token.create() -> !vm.ref<!async.token>

// Queries whether the token has been reached and returns its status.
// Returns OK if the token has been signaled successfully, DEFERRED if it is
// unsignaled, and otherwise an error indicating the failure.
vm.import private @token.query(
  %token : !vm.ref<!async.token>
) -> i32

// Signals the token.
vm.import private @token.signal(
  %token : !vm.ref<!async.token>
)

// Signals the token with a failure. The |status| will be returned from
// `async.token.query` and `async.token.await`.
vm.import private @token.fail(
  %token : !vm.ref<!async.token>,
  %status : i32
)

// Yields the caller until async token is available.
vm.import private @token.await(%token : !vm.ref<!async.token>)

//===----------------------------------------------------------------------===//
// iree_async_value_t
//===----------------------------------------------------------------------===//

vm.import private @value.await.i32(%value : !vm.ref<!async.value<i32>>) -> i32
vm.import private @value.await.i64(%value : !vm.ref<!async.value<i64>>) -> i64
vm.import private @value.await.f32(%value : !vm.ref<!async.value<f32>>) -> f32
//TODO: iree does not support return f64
//vm.import private @value.await.f64(%value : !vm.ref<!async.value<f64>>) -> f64

} // module