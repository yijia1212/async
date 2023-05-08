// IREE Hardware Abstraction Layer (HAL) runtime module imports.
//
// This is embedded in the compiler binary and inserted into any module
// containing HAL dialect ops (hal.*) that is lowered to the VM dialect.
vm.module @async_runtime {
//===----------------------------------------------------------------------===//
// iree_async_runtime_token_t
//===----------------------------------------------------------------------===//

// Returns an unavailable token
vm.import private @async.runtime.token.create() -> !vm.ref<!async.token>

// Queries whether the fence has been reached and returns its status.
// Returns OK if the fence has been signaled successfully, DEFERRED if it is
// unsignaled, and otherwise an error indicating the failure.
vm.import private @async.runtime.token.query(
  %token : !vm.ref<!async.token>
) -> i32

// Signals the fence.
vm.import private @async.runtime.token.signal(
  %token : !vm.ref<!async.token>
)

// Signals the token with a failure. The |status| will be returned from
// `async.runtime.token.query` and `async.runtime.token.await`.
vm.import private @async.runtime.token.fail(
  %token : !vm.ref<!async.token>,
  %status : i32
)

// Yields the caller until async token is available.
vm.import private @async.runtime.token.await(
  %timeout_millis : i32,
  %token : !vm.ref<!async.token> ...
) -> i32
attributes {vm.yield}

}