// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ASYNC_OPS
#define ASYNC_OPS

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

include "async/IR/AsyncDialect.td"
include "async/IR/AsyncTypes.td"

//===----------------------------------------------------------------------===//
// Async op definitions
//===----------------------------------------------------------------------===//

// Base class for the operation in this dialect
class Async_Op<string mnemonic, list<Trait> traits = []> :
    Op<AsyncDialect, mnemonic, traits>;

def Async_AwaitOp : Async_Op<"await"> {
  let summary = "waits for the argument to become ready";
  let description = [{
    The `async.await` operation waits until the argument becomes ready, and for
    the `async.value` arguments it unwraps the underlying value

    Example:

    ```mlir
    %0 = ... : !async.token
    async.await %0 : !async.token

    %1 = ... : !async.value<f32>
    %2 = async.await %1 : !async.value<f32>
    ```
  }];

  let arguments = (ins Async_AnyValueOrTokenType:$operand);
  let results = (outs Optional<AnyType>:$result);

  let skipDefaultBuilders = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Value":$operand,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
  ];

  let extraClassDeclaration = [{
    std::optional<Type> getResultType() {
      if (getResultTypes().empty()) return std::nullopt;
      return getResultTypes()[0];
    }
  }];

  let assemblyFormat = [{
    $operand `:` custom<AwaitResultType>(
      type($operand), type($result)
    ) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Async Dialect Runtime Operations.
//===----------------------------------------------------------------------===//

// The following operations are intermediate async dialect operations to help
// lowering from high level async operation like `async.execute` to the Async
// Runtime API defined in the `ExecutionEngine/AsyncRuntime.h`.

def Async_RuntimeCreateOp : Async_Op<"runtime.create"> {
  let summary = "creates an async runtime token or value";
  let description = [{
    The `async.runtime.create` operation creates an async dialect token or
    value. Tokens and values are created in the non-ready state.
  }];

  let results = (outs Async_AnyValueOrTokenType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Async_RuntimeSetAvailableOp : Async_Op<"runtime.set_available"> {
  let summary = "switches token or value to available state";
  let description = [{
    The `async.runtime.set_available` operation switches async token or value
    state to available.
  }];

  let arguments = (ins Async_AnyValueOrTokenType:$operand);
  let assemblyFormat = "$operand attr-dict `:` type($operand)";
}

def Async_RuntimeSetErrorOp : Async_Op<"runtime.set_error"> {
  let summary = "switches token or value to error state";
  let description = [{
    The `async.runtime.set_error` operation switches async token or value
    state to error.
  }];

  let arguments = (ins Async_AnyValueOrTokenType:$operand);
  let assemblyFormat = "$operand attr-dict `:` type($operand)";
}

def Async_RuntimeIsErrorOp : Async_Op<"runtime.is_error"> {
  let summary = "returns true if token, value or group is in error state";
  let description = [{
    The `async.runtime.is_error` operation returns true if the token, value or
    group (any of the async runtime values) is in the error state. It is the
    caller responsibility to check error state after the call to `await` or
    resuming after `await_and_resume`.
  }];

  let arguments = (ins Async_AnyAsyncType:$operand);
  let results = (outs I1:$is_error);

  let assemblyFormat = "$operand attr-dict `:` type($operand)";
}

def Async_RuntimeAwaitAndResumeOp : Async_Op<"runtime.await_and_resume"> {
  let summary = "awaits the async operand and resumes the coroutine";
  let description = [{
    The `async.runtime.await_and_resume` operation awaits for the operand to
    become available or error and resumes the coroutine on a thread managed by
    the runtime.
  }];

  let arguments = (ins Async_AnyAsyncType:$operand,
                       Async_CoroHandleType:$handle);
  let assemblyFormat = "$operand `,` $handle attr-dict `:` type($operand)";
}

#endif // ASYNC_OPS