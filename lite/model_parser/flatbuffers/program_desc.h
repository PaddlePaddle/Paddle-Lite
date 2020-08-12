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

class ProgramDescView : public ProgramDescAPI {
 public:
  ProgramDescView() = default;
  explicit ProgramDescView(const std::vector<char>& buf) { Init(buf); }
  explicit ProgramDescView(std::vector<char>&& buf) {
    Init(std::forward<std::vector<char>>(buf));
  }

  void Init(const std::vector<char>& buf) {
    CHECK(buf.data());
    buf_ = buf;
    InitProgramDesc();
  }

  void Init(std::vector<char>&& buf) {
    CHECK(buf.data());
    buf_ = std::move(buf);
    InitProgramDesc();
  }

  void InitProgramDesc() {
    desc_ = proto::GetProgramDesc(buf_.data());
    blocks_.resize(BlocksSize());
    for (size_t idx = 0; idx < BlocksSize(); ++idx) {
      blocks_[idx] = BlockDescView(desc_->blocks()->Get(idx));
    }
  }

  void CopyFrom(const ProgramDescView& other) {
    buf_ = other.buf();
    Init(buf_);
  }

  size_t BlocksSize() const override { return desc_->blocks()->size(); }

  template <typename T>
  T const* GetBlock(int32_t idx) const;

  template <typename T>
  T* GetBlock(int32_t idx) {
    NotImplemented();
    return nullptr;
  }

  const std::vector<BlockDescView>& GetBlocks() const { return blocks_; }

  bool HasVersion() const override { return desc_->version() != nullptr; }

  int64_t Version() const override {
    CHECK(HasVersion());
    return desc_->version()->version();
  }

  proto::ProgramDesc const* raw_desc() const { return desc_; }

  const std::vector<char>& buf() const { return buf_; }

 private:
  proto::ProgramDesc const* desc_;
  std::vector<char> buf_;
  std::vector<BlockDescView> blocks_;

 private:
  ProgramDescView& operator=(const ProgramDescView&) = delete;
  ProgramDescView(const ProgramDescView&) = delete;
  void NotImplemented() const {
    LOG(FATAL) << "The additional interfaces of ProgramDescView is temporarily "
                  "unavailable in read-only mode.";
  }
};

class ProgramDesc : public ProgramDescAPI {
 public:
  ProgramDesc() = default;

  explicit ProgramDesc(const std::vector<char>& buf) {
    const auto* raw_buf = proto::GetProgramDesc(buf.data());
    raw_buf->UnPackTo(&desc_);
    SyncBlocks();
  }

  size_t BlocksSize() const override { return desc_.blocks.size(); }

  void ClearBlocks() override {
    desc_.blocks.clear();
    SyncBlocks();
  }

  template <typename T>
  T* GetBlock(int32_t idx);

  template <typename T>
  T* AddBlock();

  bool HasVersion() const override { return desc_.version.get(); }

  int64_t Version() const override {
    if (!HasVersion()) {
      return -1;
    }
    return desc_.version->version;
  }

  void SetVersion(int64_t version_in) override {
    if (!HasVersion()) {
      desc_.version.reset(new fbs::proto::VersionT());
    }
    desc_.version->version = version_in;
  }

  const void* data() {
    SyncBuffer();
    return buf_.data();
  }

  size_t buf_size() {
    SyncBuffer();
    return buf_.size();
  }

 private:
  void SyncBlocks() {
    blocks_.resize(desc_.blocks.size());
    for (size_t i = 0; i < desc_.blocks.size(); ++i) {
      if (!blocks_[i] || blocks_[i]->raw_desc() != desc_.blocks[i].get()) {
        blocks_[i].reset(new BlockDesc(desc_.blocks[i].get()));
      }
    }
  }

  void SyncBuffer() {
    fbb_.Reset();
    flatbuffers::Offset<proto::ProgramDesc> desc =
        proto::ProgramDesc::Pack(fbb_, &desc_);
    fbb_.Finish(desc);
    buf_ = fbb_.Release();
  }

  flatbuffers::DetachedBuffer buf_;
  flatbuffers::FlatBufferBuilder fbb_;
  proto::ProgramDescT desc_;
  std::vector<std::unique_ptr<BlockDesc>> blocks_;
};

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
