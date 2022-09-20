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

#include <gtest/gtest.h>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/test/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

class MeshgridComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::vector<std::string> x_name_;
  std::vector<std::string> outs_name_;
  DDim x_dims_{{3, 5, 4, 4}};
  int x_num_ = 3;

 public:
  MeshgridComputeTester(const Place& place,
                        const std::string& alias,
                        const DDim x_dims,
                        const int x_num = 1)
      : TestCase(place, alias), x_dims_(x_dims), x_num_(x_num) {
    for (int i = 0; i < x_num; i++) {
      outs_name_.push_back("Out_" + std::to_string(i));
    }
  }

  void RunBaseline(Scope* scope) override {
    std::vector<const Tensor*> ins;
    for (std::string& name : x_name_) {
      ins.push_back(scope->FindTensor(name));
    }

    std::vector<Tensor*> outs;
    for (auto out_name : outs_name_) {
      auto* out = scope->NewTensor(out_name);
      outs.push_back(out);
    }
    int64_t size = ins.size();
    std::vector<int64_t> shape(size);
    for (int64_t i = 0; i < size; ++i) {
      switch (ins[i]->dims().size()) {
        case 0:
          shape[i] = 1;
          break;
        case 1:
          shape[i] = ins[i]->dims()[0];
          break;
        default:
          LOG(FATAL) << "Meshgrid Op expected scalar or 1D tensor in the input "
                        "tensor list";
          break;
      }
    }
    DDim out_dims;
    out_dims.ConstructFrom(shape);
    for (int64_t i = 0; i < size; i++) {
      float* dst = outs[i]->template mutable_data<float>();
      outs[i]->Resize(out_dims);
      Tensor reshape_ins_tensor;
      reshape_ins_tensor.ShareDataWith(*ins[i]);
      std::vector<int64_t> view_shape(size, 1);
      view_shape[i] = shape[i];
      DDim in_dims_reshape;
      in_dims_reshape.ConstructFrom(view_shape);
      reshape_ins_tensor.Resize(in_dims_reshape);
      const float* src = reshape_ins_tensor.data<float>();
      std::vector<int> bcast_dims(size);
      for (int64_t j = 0; j < size; j++) {
        bcast_dims[j] = shape[j];
      }
      bcast_dims[i] = 1;
      int inner_num = 1;
      int idx = size - 1;
      int outer_num = in_dims_reshape.count(0, idx);
      inner_num *= in_dims_reshape[idx];
      for (int j = 0; j < outer_num; ++j) {
        for (int k = 0; k < bcast_dims[idx]; ++k) {
          memcpy(dst + (j * bcast_dims[idx] + k) * inner_num,
                 src + j * inner_num,
                 sizeof(float) * inner_num);
        }
      }
      inner_num *= bcast_dims[idx];
      for (int idx = size - 2; idx >= 0; --idx) {
        int outer_num = in_dims_reshape.count(0, idx);
        inner_num *= in_dims_reshape[idx];
        for (int j = outer_num - 1; j >= 0; --j) {
          for (int k = bcast_dims[idx] - 1; k >= 0; --k) {
            memcpy(dst + (j * bcast_dims[idx] + k) * inner_num,
                   dst + j * inner_num,
                   sizeof(float) * inner_num);
          }
        }
        inner_num *= bcast_dims[idx];
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("meshgrid");
    op_desc->SetInput("X", x_name_);
    op_desc->SetOutput("Out", outs_name_);
  }

  void PrepareData() override {
    for (int n = 0; n < x_num_; n++) {
      std::vector<float> x_data(x_dims_.production());
      fill_data_rand(x_data.data(), -1.f, 1.f, x_dims_.production());
      const std::string x_name = "X_" + paddle::lite::to_string(n);
      x_name_.push_back(x_name);
      SetCommonTensor(x_name, x_dims_, x_data.data());
    }
  }
};

void TestMeshgrid(Place place, float abs_error, const std::string& alias) {
  DDimLite x_dims{{5}};
  std::unique_ptr<arena::TestCase> tester(
      new MeshgridComputeTester(place, alias, x_dims, 1));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

TEST(meshgrid, precision) {
  Place place;
  float abs_error = 1e-5;
  std::string alias = "float32";
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
  alias = "def";
#else
  return;
#endif
#elif defined(LITE_WITH_XPU)
  place = TARGET(kXPU);
#elif defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif

  TestMeshgrid(place, abs_error, alias);
}

}  // namespace lite
}  // namespace paddle
