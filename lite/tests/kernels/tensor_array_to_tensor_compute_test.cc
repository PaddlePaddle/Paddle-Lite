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

#include <gtest/gtest.h>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/backends/host/math/concat.h"
#include "lite/backends/host/math/stack.h"
#include "lite/core/test/arena/framework.h"

namespace paddle {
namespace lite {

DDim infer_shape(const std::vector<const Tensor*>& inputs,
                 int axis,
                 bool use_stack) {
  const size_t n = inputs.size();
  if (use_stack) {
    auto input_dims = inputs[0]->dims();
    int rank = input_dims.size();
    if (axis < 0) axis += (rank + 1);
    auto vec = input_dims.Vectorize();
    vec.insert(vec.begin() + axis, inputs.size());
    return DDimLite(vec);
  } else {
    auto out_dims = inputs[0]->dims();
    size_t in_zero_dims_size = out_dims.size();
    for (size_t i = 1; i < n; i++) {
      const auto& input_dims_i = inputs[i]->dims();
      for (size_t j = 0; j < in_zero_dims_size; j++) {
        if (j == static_cast<size_t>(axis)) {
          out_dims[axis] += input_dims_i[j];
        }
      }
    }
    if (out_dims[axis] < 0) {
      out_dims[axis] = -1;
    }
    return out_dims;
  }
}

class TensorArrayToTensorComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "x";
  std::string out_ = "out";
  std::string OutIndex_ = "outIndex";
  int axis_ = 0;
  bool use_stack_ = false;

  int x_num_ = 3;
  DDim x_dims_{{2, 3, 4, 5}};

 public:
  TensorArrayToTensorComputeTester(const Place& place,
                                   const std::string& alias,
                                   int axis,
                                   bool use_stack)
      : TestCase(place, alias) {
    axis_ = axis;
    use_stack_ = use_stack;
  }

  void RunBaseline(Scope* scope) override {
    auto x = scope->FindVar(x_)->GetMutable<std::vector<Tensor>>();
    std::vector<const Tensor*> x_vct;
    for (int i = 0; i < x->size(); i++) {
      x_vct.push_back(&(*x)[i]);
    }

    auto out = scope->NewTensor(out_);
    auto outIndex = scope->NewTensor(OutIndex_);
    CHECK(out);
    CHECK(outIndex);

    DDim output_dims = infer_shape(x_vct, axis_, use_stack_);
    out->Resize(output_dims);

    int axis = axis_;
    auto index_dim = outIndex->dims();
    if (index_dim.size() == 0) {
      std::vector<int64_t> index;
      index.push_back(x_vct.size());
      index_dim.ConstructFrom(index);
    } else {
      index_dim[0] = x_vct.size();
    }
    outIndex->Resize(index_dim);
    auto OutIndex_data = outIndex->mutable_data<float>();
    int n = x_vct.size();

    for (int i = 0; i < n; i++) {
      auto& input_dims_i = x_vct[i]->dims();
      OutIndex_data[i] = input_dims_i[axis];
    }

    auto* y_data = out->mutable_data<float>();
    // auto *y_data = out->mutable_data<float>();

    if (use_stack_) {
      if (axis < 0) axis += (x_vct[0]->dims().size() + 1);

      int pre = 1, post = 1;
      auto& dim = x_vct[0]->dims();
      for (auto i = 0; i < axis; ++i) pre *= dim[i];
      for (auto i = axis; i < dim.size(); ++i) post *= dim[i];

      auto* in_data = x_vct[0]->data<float>();
      size_t x_offset = 0;
      size_t y_offset = 0;

      for (int i = 0; i < pre; i++) {
        for (int j = 0; j < n; j++) {
          in_data = x_vct[j]->data<float>();
          for (int k = 0; k < post; k++) {
            y_data[y_offset + k] = in_data[k + x_offset];
          }
          y_offset += post;
        }
        x_offset += post;
      }

    } else {
      size_t num = x_vct.size();
      auto dim_0 = x_vct[0]->dims();
      int64_t concat_input_size = 1;
      int64_t num_cancats = 1;
      for (int i = axis + 1; i < dim_0.size(); i++) {
        concat_input_size *= dim_0[i];
      }
      for (int i = 0; i < axis; i++) {
        num_cancats *= dim_0[i];
      }

      auto* dst_ptr = out->mutable_data<float>();
      const int out_concat_axis = out->dims()[axis];
      int64_t offset_concat_axis = 0;
      int64_t out_sum = out_concat_axis * concat_input_size;
      for (int n = 0; n < num; n++) {
        auto dims = x_vct[n]->dims();
        auto* src_ptr = x_vct[n]->data<float>();
        int64_t in_concat_axis = dims[axis];
        auto* dout_ptr = dst_ptr + offset_concat_axis * concat_input_size;
        int64_t in_sum = in_concat_axis * concat_input_size;
        for (int i = 0; i < num_cancats; i++) {
          std::memcpy(dout_ptr, src_ptr, sizeof(float) * in_sum);
          dout_ptr += out_sum;
          src_ptr += in_sum;
        }
        offset_concat_axis += in_concat_axis;
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("tensor_array_to_tensor");
    op_desc->SetInput("X", {x_});
    op_desc->SetAttr("axis", axis_);
    op_desc->SetAttr("use_stack", use_stack_);
    op_desc->SetOutput("Out", {out_});
    op_desc->SetOutput("OutIndex", {OutIndex_});
  }

  void PrepareData() override {
    std::vector<std::vector<float>> x_data(x_num_);
    std::vector<DDim> dims(x_num_);

    for (int n = 0; n < x_num_; n++) {
      dims[n] = x_dims_;
      x_data[n].resize(x_dims_.production());
      for (int i = 0; i < x_dims_.production(); i++) {
        x_data[n][i] = static_cast<float>(i + n);
      }
    }
    SetCommonTensorList(x_, dims, x_data);
  }
};

TEST(TensorArrayToTensor, precision) {
  LOG(INFO) << "test TensorArrayToTensor op, kHost";
  Place place;
  float abs_error = 2e-5;

#if defined(LITE_WITH_ARM)
  place = TARGET(kHost);
#else
  return;
#endif

  for (int axis : {0, 1, 2, 3}) {
    for (bool use_stack : {true, false}) {
      std::unique_ptr<arena::TestCase> tester(
          new TensorArrayToTensorComputeTester(place, "def", axis, use_stack));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision();
    }
  }
}

}  // namespace lite
}  // namespace paddle
