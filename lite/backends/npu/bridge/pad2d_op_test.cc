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

#include "lite/operators/pad2d_op.h"
#include <gtest/gtest.h>
#include "lite/backends/npu/bridge/registry.h"
#include "lite/backends/npu/bridge/test_helper.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

template <typename dtype>
void pad2d_ref(const std::shared_ptr<operators::Pad2dOpLite> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();
  auto x = scope->FindMutableTensor(op_info->Input("X").front());
  auto out = scope->FindMutableTensor(op_info->Output("Out").front());

  auto paddings = op_info->GetAttr<std::vector<int>>("paddings");
  int pad_top = paddings[0];
  int pad_bottom = paddings[1];
  int pad_left = paddings[2];
  int pad_right = paddings[3];

  auto mode = op_info->GetAttr<std::string>("mode");
  int pad_mode;
  if (mode == "constant") {
    pad_mode = 0;
  } else if (mode == "reflect") {
    pad_mode = 1;
  } else if (mode == "edge") {
    pad_mode = 2;
  } else {
    LOG(FATAL) << "Unknown mode type";
  }
  float pad_value = op_info->GetAttr<float>("pad_value");

  auto out_dims = out->dims();
  int n = out_dims[0];
  int c = out_dims[1];
  int h = out_dims[2];
  int w = out_dims[3];

  int in_w = w - pad_left - pad_right;
  int in_h = h - pad_bottom - pad_top;
  int spatial_size_out = w * h;
  int spatial_size_in = in_w * in_h;

  auto x_data = x->data<float>();
  auto out_data = out->mutable_data<float>();
#pragma omp parallel for
  for (int i = 0; i < n * c; ++i) {
    const float* din_batch = x_data + i * spatial_size_in;
    float* dout_batch = out_data + i * spatial_size_out;
    int in_y = 0;
    int in_x = 0;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        switch (pad_mode) {
          case 0:
            in_y = y - pad_top;
            in_x = x - pad_left;
            dout_batch[y * w + x] =
                (in_x >= 0 && in_x < in_w) && (in_y >= 0 && in_y < in_h)
                    ? din_batch[in_y * in_w + in_x]
                    : pad_value;
            break;
          case 1:
            in_x =
                std::min(std::max(pad_left, x), in_w + pad_left - 1) - pad_left;
            in_y = std::min(std::max(pad_top, y), in_h + pad_top - 1) - pad_top;
            dout_batch[y * w + x] = din_batch[in_y * in_w + in_x];
            break;
          case 2:
            in_y = y - pad_top;
            in_x = x - pad_left;
            in_y = std::max(in_y, -in_y);
            in_y = std::min(in_y, 2 * in_h - in_y - 2);
            in_x = std::max(in_x, -in_x);
            in_x = std::min(in_x, 2 * in_w - in_x - 2);
            dout_batch[y * w + x] = din_batch[in_y * in_w + in_x];
            break;
          default:
            LOG(ERROR) << "ERROR: unknown pad mode:" << pad_mode;
        }
      }
    }
  }
}

void test_pad2d(int bs,
                int ic,
                int ih,
                int iw,
                std::vector<int> paddings,
                float pad_value,
                std::string mode) {
  // prepare input&output variables
  Scope scope;
  std::string x_var_name = "x";
  std::string out_var_name = "out";
  std::string out_ref_var_name = "out_ref";
  auto* x = scope.NewTensor(x_var_name);
  auto* out = scope.NewTensor(out_var_name);
  auto* out_ref = scope.NewTensor(out_ref_var_name);
  x->Resize({bs, ic, ih, iw});

  // initialize input&output data
  //  FillTensor<float, int>(x);
  auto x_data = x->mutable_data<float>();

  // initialize op desc
  cpp::OpDesc opdesc;
  opdesc.SetType("pad2d");
  opdesc.SetInput("X", {x_var_name});
  opdesc.SetOutput("Out", {out_var_name});
  opdesc.SetAttr("paddings", paddings);
  opdesc.SetAttr("pad_value", pad_value);
  opdesc.SetAttr("mode", mode);
  opdesc.SetAttr("data_format", std::string("NCHW"));

  auto op = CreateOp<operators::Pad2dOpLite>(opdesc, &scope);
  pad2d_ref<float>(op);
  out_ref->CopyDataFrom(*out);

  LauchOp(op, {x_var_name}, {out_var_name});

  // compare results
  auto* out_data = out->mutable_data<float>();
  auto* out_ref_data = out_ref->mutable_data<float>();
  for (int i = 0; i < out->numel(); i++) {
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-2) << "-----" << i;
  }
}

TEST(NPUBridges, pad2d) {
#if 1
  for (auto bs : {1, 4, 7}) {
    for (auto ic : {1, 4, 7}) {
      for (auto ih : {1, 4, 7}) {
        for (auto iw : {1, 4, 7}) {
          for (auto paddings : {/*std::vector<int>{0, 0, 0, 0},*/
                                std::vector<int>{0, 0, 0, 1},
                                std::vector<int>{0, 1, 0, 2},
                                std::vector<int>{1, 2, 3, 4}}) {
            // npu not support pad_value!=0
            for (auto pad_value : {0.f /*,1.f*/}) {
              // npu only support constant
              for (auto mode : {"constant" /*, "reflect", "edge"*/}) {
                if (mode == "edge") continue;
                VLOG(3) << "bs: " << bs << "  ic: " << ic << "  ih: " << ih
                        << "  iw: " << iw << "  paddings: {" << paddings[0]
                        << "," << paddings[1] << "," << paddings[2] << ","
                        << paddings[3] << "}"
                        << "  pad_value: " << pad_value << "  mode: " << mode;
                test_pad2d(bs, ic, ih, iw, paddings, pad_value, mode);
              }
            }
          }
        }
      }
    }
  }
#else
  test_pad2d(1, 1, 1, 1, {0, 0, 0, 1}, 0, "constant");
#endif
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(pad2d);
USE_NPU_BRIDGE(pad2d);
