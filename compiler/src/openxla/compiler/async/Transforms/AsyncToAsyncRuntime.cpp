// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/IR/Builders.h"
#include "openxla/compiler/async/Dialect/Async/IR/Async.h"
#include "openxla/compiler/async/Transforms/Passes.h"
#include "openxla/compiler/async/Transforms/async_runtime.imports.h"

#define GEN_PASS_DEF_ASYNCTOASYNCRUNTIME
#include "openxla/compiler/async/Transforms/Passes.h.inc"

using namespace mlir;

namespace openxla::compiler::async {

namespace {

class AsyncToAsyncRuntimePass
    : public ::impl::AsyncToAsyncRuntimeBase<AsyncToAsyncRuntimePass> {
public:
  AsyncToAsyncRuntimePass() = default;
  void runOnOperation() override;
};

} // namespace

namespace {
/// Lowering for `async.await` with a token operand.
class AwaitTokenOpLowering : public OpConversionPattern<AwaitOp> {
  using AwaitAdaptor = typename AwaitOp::Adaptor;
  using ImportOp = iree_compiler::IREE::VM::ImportOp;
public:
  AwaitTokenOpLowering(MLIRContext *context, SymbolTable &importSymbols)
      : OpConversionPattern<AwaitOp>(context) {
    importOp = importSymbols.lookup<ImportOp>("async.token.await");
    assert(importOp);
  }

  LogicalResult
  matchAndRewrite(AwaitOp op, typename AwaitOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (!op.getOperand().getType().template isa<TokenType>())
      return rewriter.notifyMatchFailure(op, "unsupported awaitable type");
    
    auto results = iree_compiler::rewriteToCall(op, adaptor, importOp, *this->getTypeConverter(), rewriter);
    if (!results.has_value()) return failure();
    rewriter.replaceOp(op, results.value());
    return success();
  }
private:
  ImportOp importOp;

};

}

//===----------------------------------------------------------------------===//
void AsyncToAsyncRuntimePass::runOnOperation() {
  ModuleOp module = getOperation();
  if (module.getBody()->empty()) return;

  MLIRContext *ctx = module->getContext();
  
  ModuleOp outerModuleOp, innerModuleOp;
  std::tie(outerModuleOp, innerModuleOp) =
        iree_compiler::VMConversionTarget::nestModuleForConversion(getOperation());

  (void)iree_compiler::appendImportModule(StringRef(iree_async_runtime_imports_create()->data,
                                     iree_async_runtime_imports_create()->size),
                             innerModuleOp);
  
  SymbolTable importSymbols(innerModuleOp);

  RewritePatternSet asyncPatterns(ctx);
  asyncPatterns.add<AwaitTokenOpLowering>(ctx, importSymbols);

  ConversionTarget runtimeTarget(*ctx);
  runtimeTarget.addLegalDialect<AsyncDialect, func::FuncDialect>();
  runtimeTarget.addIllegalOp<AwaitOp>();

  if (failed(applyPartialConversion(outerModuleOp, runtimeTarget,
                                    std::move(asyncPatterns)))) {
    signalPassFailure();
    return;
  }
}


std::unique_ptr<OperationPass<ModuleOp>> createAsyncToAsyncRuntimePass() {
  return std::make_unique<AsyncToAsyncRuntimePass>();
}

} // namespace openxla::compiler::async