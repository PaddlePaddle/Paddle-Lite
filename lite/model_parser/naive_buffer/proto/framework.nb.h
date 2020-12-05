// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "lite/model_parser/naive_buffer/naive_buffer.h"

namespace paddle {
namespace lite {
namespace naive_buffer {
namespace proto {

// Struct for framework
class OpDesc : public StructBuilder {
 public:
  // Move AttrType in OpDesc in NaiveBuffer
  enum AttrType {
    INT = 0,
    FLOAT,
    STRING,
    INTS,
    FLOATS,
    STRINGS,
    BOOLEAN,
    BOOLEANS,
    BLOCK,
    LONG,
    BLOCKS,
    LONGS
  };

  class Attr : public StructBuilder {
   public:
    explicit Attr(BinaryTable* table) : StructBuilder(table) {
      using enum_builder = EnumBuilder<AttrType>;

      NewStr("name");
      New<enum_builder>("type");
      NewInt32("i");
      NewFloat32("f");
      NewStr("s");
      New<ListBuilder<Int32Builder>>("ints");
      New<ListBuilder<Float32Builder>>("floats");
      New<ListBuilder<StringBuilder>>("strings");
      New<BoolBuilder>("b");
      New<ListBuilder<BoolBuilder>>("bools");
      NewInt32("block_idx");
      NewInt64("l");
      New<ListBuilder<Int32Builder>>("blocks_idx");
      New<ListBuilder<Int64Builder>>("longs");
    }
  };

  class Var : public StructBuilder {
   public:
    explicit Var(BinaryTable* table) : StructBuilder(table) {
      NewStr("parameter");
      New<ListBuilder<StringBuilder>>("arguments");
    }
  };

  explicit OpDesc(BinaryTable* table) : StructBuilder(table) {
    NewStr("type");
    New<ListBuilder<Var>>("inputs");
    New<ListBuilder<Var>>("outputs");
    New<ListBuilder<Attr>>("attrs");
    NewBool("is_target", false);
  }
};

enum VarDataType {
  // Pod Types
  BOOL = 0,
  INT16,
  INT32,
  INT64,
  FP16,
  FP32,
  FP64,
  // Tensor<size_t> is used in C++.
  SIZE_T,
  UINT8,
  INT8,

  // Other types that may need additional descriptions
  LOD_TENSOR,
  SELECTED_ROWS,
  FEED_MINIBATCH,
  FETCH_LIST,
  STEP_SCOPES,
  LOD_RANK_TABLE,
  LOD_TENSOR_ARRAY,
  PLACE_LIST,
  READER,
  // Any runtime decided variable type is raw
  // raw variables should manage their own allocations
  // in operators like nccl_op
  RAW,
  TUPLE
};

class TensorDesc : public StructBuilder {
 public:
  using enum_builder = EnumBuilder<VarDataType>;
  explicit TensorDesc(BinaryTable* table) : StructBuilder(table) {
    // Should only be PODType. Is enforced in C++
    New<enum_builder>("data_type");
    New<ListBuilder<Int64Builder>>("dims");
  }
};

class LoDTensorDesc : public StructBuilder {
 public:
  explicit LoDTensorDesc(BinaryTable* table) : StructBuilder(table) {
    New<TensorDesc>("tensor");
    NewInt32("lod_level", 0);
  }
};

class LoDTensorArrayDesc : public StructBuilder {
 public:
  explicit LoDTensorArrayDesc(BinaryTable* table) : StructBuilder(table) {
    New<TensorDesc>("tensor");
    NewInt32("lod_level", 0);
  }
};

class VarType : public StructBuilder {
 public:
  using Type = VarDataType;
  using enum_builder = EnumBuilder<Type>;
  using ReaderDesc = ListBuilder<LoDTensorDesc>;
  using Tuple = ListBuilder<enum_builder>;

  explicit VarType(BinaryTable* table) : StructBuilder(table) {
    New<enum_builder>("type");
    New<TensorDesc>("selected_rows");
    New<LoDTensorDesc>("lod_tensor");
    New<LoDTensorArrayDesc>("tensor_array");
    New<ReaderDesc>("reader");
    New<Tuple>("tuple");
  }
};

class VarDesc : public StructBuilder {
 public:
  explicit VarDesc(BinaryTable* table) : StructBuilder(table) {
    NewStr("name");
    New<VarType>("type");
    NewBool("persistable", false);
  }
};

class BlockDesc : public StructBuilder {
 public:
  explicit BlockDesc(BinaryTable* table) : StructBuilder(table) {
    NewInt32("idx");
    NewInt32("parent_idx");
    New<ListBuilder<VarDesc>>("vars");
    New<ListBuilder<OpDesc>>("ops");
    NewInt32("forward_block_idx", -1);
  }
};

class OpVersionMap : public StructBuilder {
  // op_version_map is not implemented on naive_buffer as
  // it's not useful in inference period.
};

class ProgramDesc : public StructBuilder {
 public:
  explicit ProgramDesc(BinaryTable* table) : StructBuilder(table) {
    New<ListBuilder<BlockDesc>>("blocks");
    NewInt64("version", 0);
  }
};

class ParamDesc : public StructBuilder {
 public:
  using lod_type = ListBuilder<ListBuilder<UInt64Builder>>;
  explicit ParamDesc(BinaryTable* table) : StructBuilder(table) {
    NewStr("name");
    NewUInt32("model_version");
    NewUInt64("lod_level");
    New<lod_type>("lod");
    NewUInt32("tensor_version");
    New<TensorDesc>("tensor_desc");
    New<PrimaryListBuilder<char>>("data");
  }
};

using CombinedParamsDesc = ListBuilder<ParamDesc>;

}  // namespace proto
}  // namespace naive_buffer
}  // namespace lite
}  // namespace paddle
