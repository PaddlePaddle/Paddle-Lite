// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <map>
#include <memory>
#include <string>

namespace paddle {
namespace lite {
namespace general {
namespace ssa {

class BlockOpProto {
 public:
  BlockOpProto(const std::string& attr_key,
               const std::string& in_key,
               const std::string& out_key)
      : attr_key_{attr_key}, in_key_{in_key}, out_key_{out_key} {}
  virtual ~BlockOpProto() = default;
  const std::string& AttrKey() const { return attr_key_; }
  const std::string& InKey() const { return in_key_; }
  const std::string& OutKey() const { return out_key_; }

 protected:
  std::string attr_key_;
  std::string in_key_;
  std::string out_key_;
};

class WhileOpProto : public BlockOpProto {
 public:
  WhileOpProto() : BlockOpProto("sub_block", "X", "Out") {}
};

class FakeBlockOpProto : public BlockOpProto {
 public:
  FakeBlockOpProto() : BlockOpProto("sub_block", "X", "Out") {}
};

class ConditionalBlockOpProto : public BlockOpProto {
 public:
  ConditionalBlockOpProto() : BlockOpProto("sub_block", "Input", "Out") {}
};

// In order to modify the block operator, we need to know the specific
// input name. Because its format is not uniform, so register here.

class BlockOpProtoRegistry {
 public:
  static BlockOpProtoRegistry& instance() {
    static BlockOpProtoRegistry instance_;
    return instance_;
  }
  BlockOpProtoRegistry() {
    protos_["while"].reset(new WhileOpProto);
    protos_["fake_block_op"].reset(new FakeBlockOpProto);
    protos_["conditional_block"].reset(new ConditionalBlockOpProto);
  }
  const std::shared_ptr<BlockOpProto>& GetProto(const std::string& op_type) {
    return protos_.at(op_type);
  }

 private:
  std::map<std::string, std::shared_ptr<BlockOpProto>> protos_;
};

}  // namespace ssa
}  // namespace general
}  // namespace lite
}  // namespace paddle
