// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/Conversion/StandardToVM/ConvertStandardToVM.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VM/Conversion/UtilToVM/ConvertUtilToVM.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/IR/Builders.h"
#include "openxla/compiler/async/Dialect/Async/IR/Async.h"
#include "openxla/compiler/async/Transforms/Passes.h"
#include "openxla/compiler/async/Transforms/async_runtime.imports.h"

#define GEN_PASS_DEF_ASYNCTOASYNCRUNTIME
#include <iostream>

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

}  // namespace

namespace {
/// Lowering for `async.await` with a token operand.
class AwaitTokenOpLowering : public OpConversionPattern<AwaitOp> {
  using AwaitAdaptor = typename AwaitOp::Adaptor;
  using ImportOp = iree_compiler::IREE::VM::ImportOp;

 public:
  AwaitTokenOpLowering(MLIRContext *context, SymbolTable &importSymbols,
                       TypeConverter &typeConverter)
      : OpConversionPattern<AwaitOp>(typeConverter, context) {
    importOp = importSymbols.lookup<ImportOp>("async.token.await");
    assert(importOp);
  }

  LogicalResult matchAndRewrite(
      AwaitOp op, typename AwaitOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!op.getOperand().getType().template isa<TokenType>()) {
      return rewriter.notifyMatchFailure(op, "unsupported awaitable type");
    }
    auto results = iree_compiler::rewriteToCall(
        op, adaptor, importOp, *this->getTypeConverter(), rewriter);
    if (!results.has_value()) return failure();
    rewriter.replaceOp(op, results.value());
    return success();
  }

 private:
  mutable ImportOp importOp;
};

class AwaitValueOpLowering : public OpConversionPattern<AwaitOp> {
  using AwaitAdaptor = typename AwaitOp::Adaptor;
  using ImportOp = iree_compiler::IREE::VM::ImportOp;

 public:
  AwaitValueOpLowering(MLIRContext *context, SymbolTable &importSymbols,
                       TypeConverter &typeConverter)
      : OpConversionPattern<AwaitOp>(typeConverter, context),
        importSymbols(importSymbols) {}

  LogicalResult matchAndRewrite(
      AwaitOp op, typename AwaitOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!op.getOperand().getType().template isa<ValueType>()) {
      return rewriter.notifyMatchFailure(op, "unsupported awaitable type");
    }
    auto resultType = op.getResultType();
    ImportOp importOp;
    if (resultType->isInteger(32)) {
      importOp = importSymbols.lookup<ImportOp>(
          std::string(kAsyncAwaitValuePrefix) + "i32");
    } else if (resultType->isInteger(64)) {
      importOp = importSymbols.lookup<ImportOp>(
          (kAsyncAwaitValuePrefix + "i64").str());
    } else if (resultType->isF32()) {
      importOp = importSymbols.lookup<ImportOp>(
          (kAsyncAwaitValuePrefix + "f32").str());
    } else if (resultType->isF64()) {
      importOp = importSymbols.lookup<ImportOp>(
          (kAsyncAwaitValuePrefix + "f64").str());
    } else {
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for async value");
    }
    assert(importOp);
    auto results = iree_compiler::rewriteToCall(
        op, adaptor, importOp, *this->getTypeConverter(), rewriter);
    if (!results.has_value()) return failure();
    rewriter.replaceOp(op, results.value());
    return success();
  }

 private:
  static constexpr StringLiteral kAsyncAwaitValuePrefix = "async.value.await.";
  SymbolTable &importSymbols;
};

}  // namespace

//===----------------------------------------------------------------------===//
void AsyncToAsyncRuntimePass::runOnOperation() {
  if (getOperation().getBody()->empty()) return;

  auto *ctx = &getContext();

  ModuleOp outerModuleOp, innerModuleOp;
  std::tie(outerModuleOp, innerModuleOp) =
      iree_compiler::VMConversionTarget::nestModuleForConversion(
          getOperation());

  (void)iree_compiler::appendImportModule(
      StringRef(iree_async_runtime_imports_create()->data,
                iree_async_runtime_imports_create()->size),
      innerModuleOp);
  SymbolTable importSymbols(innerModuleOp);

  iree_compiler::VMConversionTarget conversionTarget(ctx);
  auto options = iree_compiler::IREE::VM::TargetOptions::FromFlags::get();
  iree_compiler::IREE::VM::TypeConverter typeConverter(options);

  RewritePatternSet asyncPatterns(ctx);
  iree_compiler::populateStandardToVMPatterns(ctx, typeConverter,
                                              asyncPatterns);
  iree_compiler::populateUtilToVMPatterns(ctx, conversionTarget, typeConverter,
                                          asyncPatterns);
  asyncPatterns.add<AwaitTokenOpLowering>(ctx, importSymbols, typeConverter);
  asyncPatterns.add<AwaitValueOpLowering>(ctx, importSymbols, typeConverter);

  if (failed(applyPartialConversion(outerModuleOp, conversionTarget,
                                    std::move(asyncPatterns)))) {
    signalPassFailure();
    return;
  }
}

std::unique_ptr<OperationPass<ModuleOp>> createAsyncToAsyncRuntimePass() {
  return std::make_unique<AsyncToAsyncRuntimePass>();
}

}  // namespace openxla::compiler::async