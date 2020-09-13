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

#include "lite/operators/slice_op.h"
#include <gtest/gtest.h>
#include <utility>
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/mlu/bridges/test_helper.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

static void slice_ref(const float* input,
                      std::vector<int64_t> in_dims,
                      std::vector<int> axes,
                      std::vector<int> starts,
                      std::vector<int> ends,
                      float* out) {
  auto out_dims = in_dims;
  std::vector<int> real_starts(in_dims.size(), 0);
  std::vector<int> real_ends(in_dims.size(), 0);
  std::vector<int> real_step(in_dims.size(), 0);
  for (size_t i = 0; i < in_dims.size(); i++) {
    real_ends[i] = in_dims[i];
  }
  for (size_t i = 0; i < axes.size(); i++) {
    int dim_value = in_dims[axes[i]];
    if (dim_value > 0) {
      int start = starts[i] < 0 ? (starts[i] + dim_value) : starts[i];
      int end = ends[i] < 0 ? (ends[i] + dim_value) : ends[i];
      start = std::max(start, 0);
      end = std::max(end, 0);
      end = std::min(end, dim_value);
      out_dims[axes[i]] = end - start;
      real_starts[axes[i]] = start;
      real_ends[axes[i]] = end;
    }
  }
  const int LEN = in_dims.size();
  int dst_step[LEN];
  for (size_t i = 0; i < in_dims.size(); ++i) {
    dst_step[i] = 1;
  }
  int src_step[LEN];
  for (size_t i = 0; i < in_dims.size(); ++i) {
    src_step[i] = 1;
  }
  int out_num = out_dims[in_dims.size() - 1];
  for (int i = in_dims.size() - 2; i >= 0; i--) {
    dst_step[i] = out_dims[i + 1] * dst_step[i + 1];
    src_step[i] = in_dims[i + 1] * src_step[i + 1];
    out_num *= out_dims[i];
  }

  for (int dst_id = 0; dst_id < out_num; dst_id++) {
    int src_id = 0;
    int index_id = dst_id;
    for (size_t j = 0; j < out_dims.size(); j++) {
      int cur_id = index_id / dst_step[j];
      index_id = index_id % dst_step[j];
      src_id += (cur_id + real_starts[j]) * src_step[j];
    }
    out[dst_id] = input[src_id];
  }
}

static void test_case(std::vector<int64_t> x_shape,
                      std::vector<int64_t> out_shape,
                      std::vector<int> starts,
                      std::vector<int> ends,
                      std::vector<int> axes) {
  Scope scope;

  std::string x_var_name = "x";
  std::string out_var_name = "out";
  auto* x = scope.NewTensor(x_var_name);
  auto* out = scope.NewTensor(out_var_name);
  x->Resize(lite::DDim(x_shape));
  out->Resize(lite::DDim(out_shape));

  auto x_data = x->mutable_data<float>();
  FillTensor<float, float>(x, 0.f, 2.f);

  cpp::OpDesc opdesc;
  opdesc.SetType("slice");
  opdesc.SetInput("Input", {x_var_name});
  opdesc.SetOutput("Out", {out_var_name});
  opdesc.SetAttr("axes", axes);
  opdesc.SetAttr("starts", starts);
  opdesc.SetAttr("ends", ends);

  std::vector<float> out_ref(out->data_size(), 0);
  slice_ref(x_data, x_shape, axes, starts, ends, out_ref.data());

  auto type_cast = [](int64_t in) { return static_cast<int>(in); };
  std::vector<int> i_dims;
  std::transform(
      x_shape.cbegin(), x_shape.cend(), std::back_inserter(i_dims), type_cast);

  auto nchw2nhwc_axis = std::move(GetAxisNCHW2NHWC<int>(x_shape.size()));

  Tensor input_x;
  input_x.Resize(x->dims());
  transpose<float>(x->mutable_data<float>(),
                   input_x.mutable_data<float>(),
                   i_dims,
                   nchw2nhwc_axis);
  x->CopyDataFrom(input_x);

  auto op = CreateOp<operators::SliceOp>(opdesc, &scope);
  LaunchOp(op, {x_var_name}, {out_var_name});

  Tensor output_trans;
  auto os = out->dims().Vectorize();
  output_trans.Resize(os);
  std::vector<int> o_dims(os.size());
  for (size_t i = 0; i < os.size(); ++i) {
    o_dims[i] = os[nchw2nhwc_axis[i]];
  }
  transpose<float>(out->mutable_data<float>(),
                   output_trans.mutable_data<float>(),
                   o_dims,
                   GetAxisNHWC2NCHW<int>(x_shape.size()));

  auto out_data = output_trans.mutable_data<float>();
  for (DDim::value_type i = 0; i < out->dims().production(); i++) {
    EXPECT_NEAR(out_ref[i], out_data[i], 1e-4);
  }
}

TEST(MLUBridges, slice) {
  /* test_case({3}, {3}, {-3}, {3}, {0}); */
  test_case({3, 4}, {3, 4}, {-3, 0}, {3, 100}, {0, 1});
  test_case({3, 4, 5}, {3, 4, 2}, {-3, 0, 2}, {3, 100, -1}, {0, 1, 2});
  test_case({3, 4, 5, 6}, {3, 4, 2, 6}, {-3, 0, 2}, {3, 100, -1}, {0, 1, 2});
  /* test_case({3, 4, 5, 6, 3}, {3, 4, 2, 6, 3}, {-3, 0, 2}, {3, 100, -1}, {0,
   * 1, 2}); */
  /* test_case({3, 4, 5, 6, 5, 2}, {3, 4, 2, 6, 5, 2}, {-3, 0, 2}, {3, 100, 1},
   * {0, 1, 2}); */
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

USE_SUBGRAPH_BRIDGE(slice, kMLU);
