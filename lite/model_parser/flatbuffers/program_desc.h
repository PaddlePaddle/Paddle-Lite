// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <utility>
#include <vector>
#include "lite/model_parser/base/program_desc.h"
#include "lite/model_parser/flatbuffers/block_desc.h"
#include "lite/model_parser/flatbuffers/framework_generated.h"
#include "lite/utils/all.h"

namespace paddle {
namespace lite {
namespace fbs {

class ProgramDesc : public ProgramDescAPI {
 public:
  ProgramDesc() = default;
  explicit ProgramDesc(std::unique_ptr<const char[]> buf) {
    Init(std::move(buf));
  }

  size_t BlocksSize() const override { return desc_->blocks()->size(); }

  void Init(std::unique_ptr<const char[]> buf) {
    CHECK(buf.get() != nullptr);
    buf_ = std::move(buf);
    desc_ = proto::GetProgramDesc(buf_.get());
    blocks_.reserve(BlocksSize());
    for (size_t idx = 0; idx < BlocksSize(); ++idx) {
      blocks_.push_back(BlockDesc(desc_->blocks()->Get(idx)));
    }
  }

  void CopyFrom(const ProgramDesc& other) {
    size_t length = strlen(static_cast<const char*>(other.raw_buf()));
    std::unique_ptr<char[]> buf(new char[length]);
    memcpy(buf.get(), other.raw_buf(), length);
    Init(std::move(buf));
  }

  template <typename T>
  T const* GetBlock(int32_t idx) const;

  template <typename T>
  T* GetBlock(int32_t idx) {
    NotImplemented();
    return nullptr;
  }

  const std::vector<BlockDesc>& GetBlocks() const { return blocks_; }

  bool HasVersion() const override { return desc_->version() != nullptr; }

  int64_t Version() const override {
    CHECK(HasVersion());
    return desc_->version()->version();
  }

  proto::ProgramDesc const* raw_desc() const { return desc_; }

  const void* raw_buf() const { return buf_.get(); }

 private:
  proto::ProgramDesc const* desc_;
  std::unique_ptr<const char[]> buf_;
  std::vector<BlockDesc> blocks_;

 private:
  ProgramDesc& operator=(const ProgramDesc&) = delete;
  ProgramDesc(const ProgramDesc&) = delete;
  void NotImplemented() const {
    LOG(FATAL) << "The additional interfaces of ProgramDesc is temporarily "
                  "unavailable in read-only mode.";
  }
};

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
