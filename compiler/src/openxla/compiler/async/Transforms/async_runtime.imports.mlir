vm.module @async {
//===----------------------------------------------------------------------===//
// iree_async_runtime_token_t
//===----------------------------------------------------------------------===//

// Returns an unavailable token
vm.import private @async.token.create() -> !vm.ref<!async.token>

// Queries whether the token has been reached and returns its status.
// Returns OK if the token has been signaled successfully, DEFERRED if it is
// unsignaled, and otherwise an error indicating the failure.
vm.import private @async.token.query(
  %token : !vm.ref<!async.token>
) -> i32

// Signals the token.
vm.import private @async.token.signal(
  %token : !vm.ref<!async.token>
)

// Signals the token with a failure. The |status| will be returned from
// `async.token.query` and `async.token.await`.
vm.import private @async.token.fail(
  %token : !vm.ref<!async.token>,
  %status : i32
)

// Yields the caller until async token is available.
vm.import private @async.token.await(
  %token : !vm.ref<!async.token> ...
)
attributes {vm.yield}

} // module