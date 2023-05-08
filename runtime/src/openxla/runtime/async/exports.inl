
// clang-format off

EXPORT_FN("token.await", iree_async_runtime_module_token_await, r, v)
EXPORT_FN("token.create", iree_async_runtime_module_token_create, v, r)
EXPORT_FN("token.fail", iree_async_runtime_module_token_fail, r, v)
EXPORT_FN("token.query", iree_async_runtime_module_token_query, r, i)
EXPORT_FN("token.signal", iree_async_runtime_module_token_signal, r, v)

// clang-format on