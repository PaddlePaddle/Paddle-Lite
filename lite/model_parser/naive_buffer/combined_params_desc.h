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

#include <algorithm>
#include <string>
#include <vector>
#include "lite/core/model/base/apis.h"
#include "lite/model_parser/naive_buffer/param_desc.h"
#include "lite/model_parser/naive_buffer/proto/framework.nb.h"

namespace paddle {
namespace lite {
namespace naive_buffer {

class CombinedParamsDesc {
 public:
  CombinedParamsDesc() = delete;

  explicit CombinedParamsDesc(proto::CombinedParamsDesc *desc) : desc_(desc) {
    CHECK(desc_);
  }

  void CopyFrom(CombinedParamsDesc &combined_params_desc) {  // NOLINT
    CHECK(combined_params_desc.Proto())
        << "Source proto::CombinedParamsDesc pointer can't be null";
    desc_ = combined_params_desc.Proto();
  }

  proto::CombinedParamsDesc *Proto() { return desc_; }

  const proto::CombinedParamsDesc &ReadonlyProto() const { return *desc_; }

  size_t ParamsSize() const { return desc_->size(); }

  void ClearParams() { desc_->Clear(); }

  proto::ParamDesc *GetParam(int32_t idx) {
    CHECK_LT(idx, ParamsSize()) << "idx >= params.size()";
    return desc_->GetMutable(idx);
  }

  proto::ParamDesc *AddParam() { return desc_->New(); }

 private:
  proto::CombinedParamsDesc *desc_;
};

}  // namespace naive_buffer
}  // namespace lite
}  // namespace paddle
