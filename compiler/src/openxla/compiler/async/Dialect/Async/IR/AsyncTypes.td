// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


#ifndef ASYNC_TYPES
#define ASYNC_TYPES

include "mlir/IR/AttrTypeBase.td"
include "openxla/compiler/async/Dialect/Async/IR/AsyncDialect.td"

//===----------------------------------------------------------------------===//
// Async Types
//===----------------------------------------------------------------------===//

class Async_Type<string name, string typeMnemonic> : TypeDef<AsyncDialect,
                                                             name> {
  let mnemonic = typeMnemonic;
}

def Async_TokenType : Async_Type<"Token", "token"> {
  let summary = "async token type";
  let description = [{
    `async.token` is a type returned by asynchronous operations, and it becomes
    `available` when the asynchronous operations that created it is completed.
  }];
}


def Async_ValueType : Async_Type<"Value", "value"> {
  let summary = "async value type";
  let description = [{
    `async.value` represents a value returned by asynchronous operations,
    which may or may not be available currently, but will be available at some
    point in the future.
  }];

  let parameters = (ins "Type":$valueType);
  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$valueType), [{
      return $_get(valueType.getContext(), valueType);
    }]>
  ];
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

def Async_AnyValueOrTokenType : AnyTypeOf<[Async_ValueType,
                                           Async_TokenType]>;

def Async_AnyAsyncType : AnyTypeOf<[Async_ValueType,
                                    Async_TokenType]>;

#endif // ASYNC_TYPES