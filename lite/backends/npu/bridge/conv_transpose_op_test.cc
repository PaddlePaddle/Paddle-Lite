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

#include "lite/operators/conv_transpose_op.h"
#include <gtest/gtest.h>
#include <random>
#include "lite/backends/npu/bridge/registry.h"
#include "lite/backends/npu/bridge/test_helper.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

template <typename DType>
void add_bias_with_relu(DType* data,
                        const DType* bias,
                        int channel_size,
                        int inner_size,
                        bool has_relu) {
  for (int c = 0; c < channel_size; ++c) {
    DType bias_val = bias != nullptr ? bias[c] : 0;
    for (int i = 0; i < inner_size; i++) {
      DType data_val = data[i];
      data_val += bias_val;
      if (has_relu) {
        data_val = data_val > 0 ? data_val : 0.f;
      }
      data[i] = data_val;
    }
    data += inner_size;
  }
}

template <typename DType>
void col2im(const DType* data_col,
            const int channel_size,
            const int height,
            const int width,
            const int kernel_h,
            const int kernel_w,
            const int pad_h,
            const int pad_w,
            const int stride_h,
            const int stride_w,
            const int dilation_h,
            const int dilation_w,
            DType* data_im) {
  memset(data_im, 0, height * width * channel_size * sizeof(DType));
  const int output_h =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int inner_size = height * width;
  for (int c = channel_size; c--; data_im += inner_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (input_row < 0 || input_row >= height) {
            data_col += output_w;
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (input_col >= 0 && input_col < width) {
                data_im[input_row * width + input_col] += *data_col;
              }
              data_col++;
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

template <typename IType, typename OType>
void gemm(int M,
          int N,
          int K,
          const IType* A,
          const IType* B,
          OType* C,
          OType alpha,
          OType beta,
          bool is_trans_A = false,
          bool is_trans_B = false) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      OType sum = static_cast<OType>(0);
      for (int k = 0; k < K; ++k) {
        IType a;
        IType b;
        if (is_trans_A) {
          a = A[k * M + m];
        } else {
          a = A[m * K + k];
        }
        if (is_trans_B) {
          b = B[n * K + k];
        } else {
          b = B[k * N + n];
        }
        sum += a * b;
      }
      C[m * N + n] = alpha * sum + beta * C[m * N + n];
    }
  }
}

template <typename IType, typename OType>
void conv_transpose_ref(
    const std::shared_ptr<operators::ConvTransposeOpLite> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();
  auto input =
      scope->FindVar(op_info->Input("Input").front())->GetMutable<Tensor>();
  auto filter =
      scope->FindVar(op_info->Input("Filter").front())->GetMutable<Tensor>();
  auto output =
      scope->FindVar(op_info->Output("Output").front())->GetMutable<Tensor>();
  std::vector<int32_t> strides =
      op_info->GetAttr<std::vector<int32_t>>("strides");
  std::vector<int32_t> paddings =
      op_info->GetAttr<std::vector<int32_t>>("paddings");
  int32_t groups = op_info->GetAttr<int32_t>("groups");
  std::vector<int32_t> dilations =
      op_info->GetAttr<std::vector<int32_t>>("dilations");
  bool fuse_relu = op_info->GetAttr<bool>("fuse_relu");
  Tensor* bias = nullptr;
  OType* bias_data = nullptr;
  if (op_info->HasInput("Bias")) {
    auto bias_var_names = op_info->Input("Bias");
    if (bias_var_names.size() > 0) {
      auto bias_var_name = bias_var_names.front();
      bias = scope->FindVar(bias_var_name)->GetMutable<Tensor>();
      bias_data = bias->mutable_data<OType>();
    }
  }
  auto input_dims = input->dims();
  auto filter_dims = filter->dims();
  auto output_dims = output->dims();
  auto input_data = input->mutable_data<IType>();
  auto filter_data = filter->mutable_data<IType>();
  auto output_data = output->mutable_data<OType>();
  int kernel_w = filter_dims[3];
  int kernel_h = filter_dims[2];
  int stride_w = strides[1];
  int stride_h = strides[0];
  int dila_w = dilations[1];
  int dila_h = dilations[0];
  int pad_w = paddings[1];
  int pad_h = paddings[0];
  int batch_size = input_dims[0];
  int in_ch_size = input_dims[1];
  int in_h = input_dims[2];
  int in_w = input_dims[3];
  int out_ch_size = output_dims[1];
  int out_h = output_dims[2];
  int out_w = output_dims[3];

  int M = out_ch_size * kernel_w * kernel_h / groups;
  int N = in_h * in_w;
  int K = in_ch_size / groups;

  if (in_ch_size != out_ch_size || groups != in_ch_size) {
    CHECK_EQ(in_ch_size % groups, 0);
    CHECK_EQ(out_ch_size % groups, 0);
  }

  auto workspace = std::vector<OType>(groups * M * N);
  int group_input_size = in_w * in_h * in_ch_size / groups;
  int group_output_size = out_w * out_h * out_ch_size / groups;
  int group_col_size = M * N;
  int group_filter_size =
      in_ch_size * out_ch_size * kernel_w * kernel_h / (groups * groups);
  bool flag_1x1s1p1 = (kernel_w == 1) && (kernel_h == 1) && (stride_h == 1) &&
                      (stride_w == 1) && (pad_w == 1) && (pad_h == 1) &&
                      (dila_w == 1) && (dila_h == 1);
  for (int n = 0; n < batch_size; ++n) {
    input_data += n * in_ch_size * in_h * in_w;
    output_data += n * out_ch_size * out_h * out_w;
    auto col_data = workspace.data();
    if (flag_1x1s1p1) {
      col_data = output_data;
    }
    memset(col_data, 0, sizeof(OType) * group_col_size);
    for (int g = 0; g < groups; ++g) {
      auto input_group_data = input_data + g * group_input_size;
      auto filter_group_data = filter_data + g * group_filter_size;
      auto col_group_data = col_data + g * group_col_size;
      gemm<IType, OType>(M,
                         N,
                         K,
                         filter_group_data,
                         input_group_data,
                         col_group_data,
                         static_cast<OType>(1),
                         static_cast<OType>(0),
                         true,
                         false);
    }
    if (!flag_1x1s1p1) {
      col2im(col_data,
             out_ch_size,
             out_h,
             out_w,
             kernel_h,
             kernel_w,
             pad_h,
             pad_w,
             stride_h,
             stride_w,
             dila_h,
             dila_w,
             output_data);
    }
    add_bias_with_relu(
        output_data, bias_data, out_ch_size, out_w * out_h, fuse_relu);
  }
}

void test_conv_transpose(int bs,
                         int ic,
                         int ih,
                         int iw,
                         bool has_bias,
                         bool fuse_relu,
                         int filters,
                         int groups,
                         int dilation,
                         int stride,
                         int padding,
                         int kernel) {
  // prepare input&output variables
  Scope scope;
  std::string input_var_name("input");
  std::string filter_var_name("filter");
  std::string bias_var_name("bias");
  std::string output_var_name("output");
  std::string output_ref_var_name("output_ref");
  auto* input = scope.Var(input_var_name)->GetMutable<Tensor>();
  auto* filter = scope.Var(filter_var_name)->GetMutable<Tensor>();
  auto* bias = scope.Var(bias_var_name)->GetMutable<Tensor>();
  auto* output = scope.Var(output_var_name)->GetMutable<Tensor>();
  auto* output_ref = scope.Var(output_ref_var_name)->GetMutable<Tensor>();

  // get group size and input&filter shape
  std::vector<int64_t> input_shape = {bs, ic, ih, iw};
  std::vector<int64_t> filter_shape = {ic, filters, kernel, kernel};
  input->Resize(input_shape);
  filter->Resize(filter_shape);

  // initialize input&output data
  FillTensor<float, int>(input);
  FillTensor<float, int>(filter);

  // initialize op desc
  cpp::OpDesc opdesc;
  opdesc.SetType("conv2d_transpose");
  opdesc.SetInput("Input", {input_var_name});
  opdesc.SetInput("Filter", {filter_var_name});
  opdesc.SetOutput("Output", {output_var_name});
  opdesc.SetAttr("dilations", std::vector<int32_t>({dilation, dilation}));
  opdesc.SetAttr("strides", std::vector<int32_t>({stride, stride}));
  opdesc.SetAttr("paddings", std::vector<int32_t>({padding, padding}));
  opdesc.SetAttr("groups", groups);
  opdesc.SetAttr("fuse_relu", static_cast<bool>(fuse_relu));
  if (has_bias) {
    bias->Resize({1, filters * groups, 1, 1});
    FillTensor<float, int>(bias);
    opdesc.SetInput("Bias", {bias_var_name});
  }

  // create and convert op to NPU model, then run it on NPU
  auto op = CreateOp<operators::ConvTransposeOpLite>(opdesc, &scope);
  LauchOp(op, {input_var_name}, {output_var_name});
  output_ref->CopyDataFrom(*output);

  // execute reference implementation and save to output tensor('out')
  conv_transpose_ref<float, float>(op);

  // compare results
  auto* output_data = output->mutable_data<float>();
  auto* output_ref_data = output_ref->mutable_data<float>();
  for (int i = 0; i < output->dims().production(); i++) {
    VLOG(5) << i;
    EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-5);
  }
}

TEST(NPUBridges, conv_transpose) {
#if 1
  for (auto bs : {1, 2}) {
    for (auto ic : {3, 6}) {
      for (auto ih : {14, 28}) {
        for (auto iw : {14, 28}) {
          for (auto has_bias : {false, true}) {
            for (auto fuse_relu : {false, true}) {
              for (auto filters : {1, 2, 5}) {
                for (auto groups : {1 /* , 2, 5*/}) {
                  for (auto dilation : {1, 2}) {
                    for (auto stride : {1, 2}) {
                      for (auto kernel : {1, 3, 5}) {
                        std::vector<int> paddings = {kernel / 2};
                        if (kernel / 2 != 0) {
                          paddings.push_back(0);
                        }
                        for (auto padding : paddings) {
                          VLOG(3) << "bs: " << bs << " ic: " << ic
                                  << " ih: " << ih << " iw: " << iw
                                  << " has_bias: " << has_bias
                                  << " fuse_relu: " << fuse_relu
                                  << " filters: " << filters
                                  << " groups: " << groups
                                  << " dilation: " << dilation
                                  << " stride: " << stride
                                  << " padding: " << padding
                                  << " kernel: " << kernel;
                          test_conv_transpose(bs,
                                              ic,
                                              ih,
                                              iw,
                                              has_bias,
                                              fuse_relu,
                                              filters,
                                              groups,
                                              dilation,
                                              stride,
                                              padding,
                                              kernel);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
#else
  test_conv_transpose(1, 6, 8, 8, false, false, 5, 2, 1, 1, 1, 3);
#endif
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(conv2d_transpose);
USE_NPU_BRIDGE(conv2d_transpose);
