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

#include <cmath>
#include <string>
#include <vector>

#include "lite/core/op_registry.h"
#include "lite/kernels/arm/elementwise_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

TEST(elementwise_add_arm, retrive_op) {
  auto elementwise_add = KernelRegistry::Global().Create("elementwise_add");
  ASSERT_FALSE(elementwise_add.empty());
  ASSERT_TRUE(elementwise_add.front());
}

TEST(elementwise_add_arm, init) {
  ElementwiseAddCompute<float, PRECISION(kFloat)> elementwise_add;
  ASSERT_EQ(elementwise_add.precision(), PRECISION(kFloat));
  ASSERT_EQ(elementwise_add.target(), TARGET(kARM));
}

template <typename dtype>
void elementwise_compute_ref(const operators::ElementwiseParam& param,
                             const std::string elt_type,
                             const std::string act_type) {
  const dtype* x_data = param.X->data<const dtype>();
  const dtype* y_data = param.Y->data<const dtype>();
  dtype* out_data = param.Out->mutable_data<dtype>();
  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  int axis = param.axis;
  if (axis < 0) {
    axis = x_dims.size() - y_dims.size();
  }
  int batch = 1;
  int channels = 1;
  int num = 1;
  for (int i = 0; i < axis; ++i) {
    batch *= x_dims[i];
  }
  for (int i = 0; i < y_dims.size(); ++i) {
    channels *= y_dims[i];
  }
  for (int i = y_dims.size() + axis; i < x_dims.size(); ++i) {
    num *= x_dims[i];
  }
  // do elementwise add/sub/max...
  if (elt_type == "add") {
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < channels; ++j) {
        int offset = (i * channels + j) * num;
        const dtype* din_ptr = x_data + offset;
        const dtype diny_data = y_data[j];
        dtype* dout_ptr = out_data + offset;
        for (int k = 0; k < num; ++k) {
          *dout_ptr = *din_ptr + diny_data;
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  } else if (elt_type == "sub") {
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < channels; ++j) {
        int offset = (i * channels + j) * num;
        const dtype* din_ptr = x_data + offset;
        const dtype diny_data = y_data[j];
        dtype* dout_ptr = out_data + offset;
        for (int k = 0; k < num; ++k) {
          *dout_ptr = *din_ptr - diny_data;
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  } else if (elt_type == "mul") {
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < channels; ++j) {
        int offset = (i * channels + j) * num;
        const dtype* din_ptr = x_data + offset;
        const dtype diny_data = y_data[j];
        dtype* dout_ptr = out_data + offset;
        for (int k = 0; k < num; ++k) {
          *dout_ptr = *din_ptr * diny_data;
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  } else if (elt_type == "div") {
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < channels; ++j) {
        int offset = (i * channels + j) * num;
        const dtype* din_ptr = x_data + offset;
        const dtype diny_data = y_data[j];
        dtype* dout_ptr = out_data + offset;
        for (int k = 0; k < num; ++k) {
          *dout_ptr = *din_ptr / diny_data;
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  } else if (elt_type == "max") {
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < channels; ++j) {
        int offset = (i * channels + j) * num;
        const dtype* din_ptr = x_data + offset;
        const dtype diny_data = y_data[j];
        dtype* dout_ptr = out_data + offset;
        for (int k = 0; k < num; ++k) {
          *dout_ptr = std::max(*din_ptr, diny_data);
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  } else {
    LOG(FATAL) << "unsupported Elementwise type: " << elt_type;
  }
  // do activation relu/sigmod...
  if (act_type.size() > 0) {
    if (act_type == "relu") {
      for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < channels; ++j) {
          dtype* dout_ptr = out_data + (i * channels + j) * num;
          for (int k = 0; k < num; ++k) {
            *dout_ptr = *dout_ptr > 0.0f ? *dout_ptr : 0.0f;
            dout_ptr++;
          }
        }
      }
    } else {
      LOG(FATAL) << "unsupported Activation type: " << elt_type;
    }
  }
}

template <typename dtype>
void elementwise_fmod_compute_ref(const operators::ElementwiseParam& param,
                                  const std::string act_type) {
  const dtype* x_data = param.X->data<const dtype>();
  const dtype* y_data = param.Y->data<const dtype>();
  dtype* out_data = param.Out->mutable_data<dtype>();
  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  int axis = param.axis;
  if (axis < 0) {
    axis = x_dims.size() - y_dims.size();
  }
  int batch = 1;
  int channels = 1;
  int num = 1;
  for (int i = 0; i < axis; ++i) {
    batch *= x_dims[i];
  }
  for (int i = 0; i < y_dims.size(); ++i) {
    channels *= y_dims[i];
  }
  for (int i = y_dims.size() + axis; i < x_dims.size(); ++i) {
    num *= x_dims[i];
  }
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      int offset = (i * channels + j) * num;
      const dtype* din_ptr = x_data + offset;
      const dtype diny_data = y_data[j];
      dtype* dout_ptr = out_data + offset;
      for (int k = 0; k < num; ++k) {
        *dout_ptr = fmod(diny_data + fmod(*din_ptr, diny_data), diny_data);
        dout_ptr++;
        din_ptr++;
      }
    }
  }
  // do activation relu
  if (act_type.size() > 0) {
    if (act_type == "relu") {
      for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < channels; ++j) {
          dtype* dout_ptr = out_data + (i * channels + j) * num;
          for (int k = 0; k < num; ++k) {
            *dout_ptr = *dout_ptr > 0.0f ? *dout_ptr : 0.0f;
            dout_ptr++;
          }
        }
      }
    }
  }
}

template <typename dtype>
void elementwise_imod_compute_ref(const operators::ElementwiseParam& param,
                                  const std::string act_type) {
  const dtype* x_data = param.X->data<const dtype>();
  const dtype* y_data = param.Y->data<const dtype>();
  dtype* out_data = param.Out->mutable_data<dtype>();
  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();
  int axis = param.axis;
  if (axis < 0) {
    axis = x_dims.size() - y_dims.size();
  }
  int batch = 1;
  int channels = 1;
  int num = 1;
  for (int i = 0; i < axis; ++i) {
    batch *= x_dims[i];
  }
  for (int i = 0; i < y_dims.size(); ++i) {
    channels *= y_dims[i];
  }
  for (int i = y_dims.size() + axis; i < x_dims.size(); ++i) {
    num *= x_dims[i];
  }
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      int offset = (i * channels + j) * num;
      const dtype* din_ptr = x_data + offset;
      const dtype diny_data = y_data[j];
      dtype* dout_ptr = out_data + offset;
      for (int k = 0; k < num; ++k) {
        *dout_ptr = (*din_ptr) % diny_data;
        dout_ptr++;
        din_ptr++;
      }
    }
  }
  // do activation relu
  if (act_type.size() > 0) {
    if (act_type == "relu") {
      for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < channels; ++j) {
          dtype* dout_ptr = out_data + (i * channels + j) * num;
          for (int k = 0; k < num; ++k) {
            *dout_ptr = *dout_ptr > 0.0f ? *dout_ptr : 0.0f;
            dout_ptr++;
          }
        }
      }
    }
  }
}

template void elementwise_fmod_compute_ref<float>(
    const operators::ElementwiseParam& param, const std::string act_type);
template void elementwise_imod_compute_ref<int32_t>(
    const operators::ElementwiseParam& param, const std::string act_type);
template void elementwise_imod_compute_ref<int64_t>(
    const operators::ElementwiseParam& param, const std::string act_type);

template <class T>
bool is_fp_close(T v1, T v2, T rel_tol = 1e-4, T abs_tol = 1e-5) {
  bool abs_chk = std::abs(v1 - v2) < abs_tol;
  bool rel_chk =
      (std::abs(v1 - v2) / std::min(std::abs(v1), std::abs(v2))) < rel_tol;
  return abs_chk || rel_chk;
}

template <template <class, PrecisionType> class ElementWiseComputeTemplate,
          typename T,
          PrecisionType PType>
void do_elementwise_compute(const char* op_type_str) {
  ElementWiseComputeTemplate<T, PType> elementwise_add;
  operators::ElementwiseParam param;
  lite::Tensor x, y, output, output_ref;
  unsigned int rand_seed = 1;

#if 1
  for (auto n : {1, 3, 4}) {
    for (auto c : {1, 3, 4}) {
      for (auto h : {1, 3, 4}) {
        for (auto w : {1, 3, 4}) {
          for (auto axis : {-1, 0, 1, 3}) {
            for (auto yd : {std::vector<int64_t>({n}),
                            std::vector<int64_t>({c}),
                            std::vector<int64_t>({h}),
                            std::vector<int64_t>({w}),
                            std::vector<int64_t>({n, c}),
                            std::vector<int64_t>({c, h}),
                            std::vector<int64_t>({c, h, w}),
                            std::vector<int64_t>({n, c, h, w})}) {
#else
  for (auto n : {1, 3, 4, 11}) {
    for (auto c : {1, 3, 4, 11}) {
      for (auto h : {1, 3, 4, 11}) {
        for (auto w : {1, 3, 4, 11}) {
          for (auto axis : {-1, 0, 1, 2, 3}) {
            for (auto yd : {std::vector<int64_t>({n}),
                            std::vector<int64_t>({c}),
                            std::vector<int64_t>({h}),
                            std::vector<int64_t>({w}),
                            std::vector<int64_t>({n, c}),
                            std::vector<int64_t>({c, h}),
                            std::vector<int64_t>({h, w}),
                            std::vector<int64_t>({n, c, h}),
                            std::vector<int64_t>({c, h, w}),
                            std::vector<int64_t>({n, c, h, w})}) {
#endif
              auto x_dim = DDim(std::vector<int64_t>({n, c, h, w}));
              auto y_dim = DDim(yd);
              int axis_t = axis < 0 ? x_dim.size() - y_dim.size() : axis;

              if (axis_t + y_dim.size() > 4) continue;
              bool flag = false;
              for (int i = 0; i < y_dim.size(); i++) {
                if (x_dim[i + axis_t] != y_dim[i]) flag = true;
              }
              if (flag) continue;

              x.Resize(x_dim);
              y.Resize(y_dim);
              output.Resize(x_dim);
              output_ref.Resize(x_dim);
              T* x_data = x.mutable_data<T>();
              T* y_data = y.mutable_data<T>();
              T* output_data = output.mutable_data<T>();
              T* output_ref_data = output_ref.mutable_data<T>();
              for (int i = 0; i < x_dim.production(); i++) {
                x_data[i] = 1.0 * rand_r(&rand_seed) * rand_r(&rand_seed) /
                            (rand_r(&rand_seed) + 1);
              }
              for (int i = 0; i < y_dim.production(); i++) {
                y_data[i] = 1.0 * rand_r(&rand_seed) * rand_r(&rand_seed) /
                            (rand_r(&rand_seed) + 1);
              }
              param.X = &x;
              param.Y = &y;
              param.axis = axis;
              param.Out = &output;
              elementwise_add.SetParam(param);
              elementwise_add.Run();
              param.Out = &output_ref;
              elementwise_compute_ref<T>(param, op_type_str, "");
              if (std::is_floating_point<T>::value) {
                for (int i = 0; i < output.dims().production(); i++) {
                  ASSERT_EQ(is_fp_close(output_data[i], output_ref_data[i]),
                            true)
                      << op_type_str << "Value differ at index " << i;
                }
              } else {
                for (int i = 0; i < output.dims().production(); i++) {
                  ASSERT_EQ(output_data[i], output_ref_data[i])
                      << op_type_str << "Value differ at index " << i;
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(elementwise_op, compute_fp32) {
  do_elementwise_compute<ElementwiseAddCompute, float, PRECISION(kFloat)>(
      "add");
  do_elementwise_compute<ElementwiseSubCompute, float, PRECISION(kFloat)>(
      "sub");
  do_elementwise_compute<ElementwiseMulCompute, float, PRECISION(kFloat)>(
      "mul");
  do_elementwise_compute<ElementwiseDivCompute, float, PRECISION(kFloat)>(
      "div");
  if (::testing::Test::HasFailure()) {
    FAIL();
  }
}

TEST(elementwise_op, compute_i32) {
  do_elementwise_compute<ElementwiseAddCompute, int32_t, PRECISION(kInt32)>(
      "add");
  do_elementwise_compute<ElementwiseSubCompute, int32_t, PRECISION(kInt32)>(
      "sub");
  do_elementwise_compute<ElementwiseMulCompute, int32_t, PRECISION(kInt32)>(
      "mul");
  do_elementwise_compute<ElementwiseDivCompute, int32_t, PRECISION(kInt32)>(
      "div");
  if (::testing::Test::HasFailure()) {
    FAIL();
  }
}

TEST(fusion_elementwise_add_activation_arm, retrive_op) {
  auto fusion_elementwise_add_activation =
      KernelRegistry::Global().Create("fusion_elementwise_add_activation");
  ASSERT_FALSE(fusion_elementwise_add_activation.empty());
  ASSERT_TRUE(fusion_elementwise_add_activation.front());
}

TEST(fusion_elementwise_add_activation_arm, init) {
  ElementwiseAddActivationCompute fusion_elementwise_add_activation;
  ASSERT_EQ(fusion_elementwise_add_activation.precision(), PRECISION(kFloat));
  ASSERT_EQ(fusion_elementwise_add_activation.target(), TARGET(kARM));
}

TEST(fusion_elementwise_add_activation_arm, compute) {
  ElementwiseAddActivationCompute fusion_elementwise_add_activation;
  operators::FusionElementwiseActivationParam param;
  lite::Tensor x, y, output, output_ref;

#if 1
  for (auto act_type : {"relu"}) {
    for (auto n : {1, 3, 4}) {
      for (auto c : {1, 3, 4}) {
        for (auto h : {1, 3, 4}) {
          for (auto w : {1, 3, 4}) {
            for (auto axis : {-1, 0, 1, 3}) {
              for (auto yd : {std::vector<int64_t>({n}),
                              std::vector<int64_t>({c}),
                              std::vector<int64_t>({h}),
                              std::vector<int64_t>({w}),
                              std::vector<int64_t>({n, c}),
                              std::vector<int64_t>({h, w}),
                              std::vector<int64_t>({n, c, h}),
                              std::vector<int64_t>({n, c, h, w})}) {
#else
  for (auto act_type : {"relu"}) {
    for (auto n : {1, 3, 4, 11}) {
      for (auto c : {1, 3, 4, 11}) {
        for (auto h : {1, 3, 4, 11}) {
          for (auto w : {1, 3, 4, 11}) {
            for (auto axis : {-1, 0, 1, 2, 3}) {
              for (auto yd : {std::vector<int64_t>({n}),
                              std::vector<int64_t>({c}),
                              std::vector<int64_t>({h}),
                              std::vector<int64_t>({w}),
                              std::vector<int64_t>({n, c}),
                              std::vector<int64_t>({c, h}),
                              std::vector<int64_t>({h, w}),
                              std::vector<int64_t>({n, c, h}),
                              std::vector<int64_t>({c, h, w}),
                              std::vector<int64_t>({n, c, h, w})}) {
#endif
                auto x_dim = DDim(std::vector<int64_t>({n, c, h, w}));
                auto y_dim = DDim(yd);
                int axis_t = axis < 0 ? x_dim.size() - y_dim.size() : axis;

                if (axis_t + y_dim.size() > 4) continue;
                bool flag = false;
                for (int i = 0; i < y_dim.size(); i++) {
                  if (x_dim[i + axis_t] != y_dim[i]) flag = true;
                }
                if (flag) continue;

                x.Resize(x_dim);
                y.Resize(y_dim);
                output.Resize(x_dim);
                output_ref.Resize(x_dim);
                auto* x_data = x.mutable_data<float>();
                auto* y_data = y.mutable_data<float>();
                auto* output_data = output.mutable_data<float>();
                auto* output_ref_data = output_ref.mutable_data<float>();
                for (int i = 0; i < x_dim.production(); i++) {
                  float sign = i % 3 == 0 ? -1.0f : 1.0f;
                  x_data[i] = i * sign;
                }
                for (int i = 0; i < y_dim.production(); i++) {
                  float sign = i % 2 == 0 ? 0.5f : -0.5f;
                  y_data[i] = i * sign;
                }
                param.X = &x;
                param.Y = &y;
                param.axis = axis;
                param.Out = &output;
                param.act_type = act_type;
                fusion_elementwise_add_activation.SetParam(param);
                fusion_elementwise_add_activation.Run();
                param.Out = &output_ref;
                elementwise_compute_ref<float>(param, "add", act_type);
                for (int i = 0; i < output.dims().production(); i++) {
                  EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-5);
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(elementwise_mul_arm, retrive_op) {
  auto elementwise_mul = KernelRegistry::Global().Create("elementwise_mul");
  ASSERT_FALSE(elementwise_mul.empty());
  ASSERT_TRUE(elementwise_mul.front());
}

TEST(elementwise_mul_arm, init) {
  ElementwiseMulCompute<float, PRECISION(kFloat)> elementwise_mul;
  ASSERT_EQ(elementwise_mul.precision(), PRECISION(kFloat));
  ASSERT_EQ(elementwise_mul.target(), TARGET(kARM));
}

TEST(elementwise_mul, compute) {
  ElementwiseMulCompute<float, PRECISION(kFloat)> elementwise_mul;
  operators::ElementwiseParam param;
  lite::Tensor x, y, output, output_ref;

#if 1
  for (auto n : {1, 3, 4}) {
    for (auto c : {1, 3, 4}) {
      for (auto h : {1, 3, 4}) {
        for (auto w : {1, 3, 4}) {
          for (auto axis : {-1, 0, 1, 3}) {
            for (auto yd : {std::vector<int64_t>({n}),
                            std::vector<int64_t>({c}),
                            std::vector<int64_t>({h}),
                            std::vector<int64_t>({w}),
                            std::vector<int64_t>({n, c}),
                            std::vector<int64_t>({c, h}),
                            std::vector<int64_t>({c, h, w}),
                            std::vector<int64_t>({n, c, h, w})}) {
#else
  for (auto n : {1, 3, 4, 11}) {
    for (auto c : {1, 3, 4, 11}) {
      for (auto h : {1, 3, 4, 11}) {
        for (auto w : {1, 3, 4, 11}) {
          for (auto axis : {-1, 0, 1, 2, 3}) {
            for (auto yd : {std::vector<int64_t>({n}),
                            std::vector<int64_t>({c}),
                            std::vector<int64_t>({h}),
                            std::vector<int64_t>({w}),
                            std::vector<int64_t>({n, c}),
                            std::vector<int64_t>({c, h}),
                            std::vector<int64_t>({h, w}),
                            std::vector<int64_t>({n, c, h}),
                            std::vector<int64_t>({c, h, w}),
                            std::vector<int64_t>({n, c, h, w})}) {
#endif
              auto x_dim = DDim(std::vector<int64_t>({n, c, h, w}));
              auto y_dim = DDim(yd);
              int axis_t = axis < 0 ? x_dim.size() - y_dim.size() : axis;

              if (axis_t + y_dim.size() > 4) continue;
              bool flag = false;
              for (int i = 0; i < y_dim.size(); i++) {
                if (x_dim[i + axis_t] != y_dim[i]) flag = true;
              }
              if (flag) continue;

              x.Resize(x_dim);
              y.Resize(y_dim);
              output.Resize(x_dim);
              output_ref.Resize(x_dim);
              auto* x_data = x.mutable_data<float>();
              auto* y_data = y.mutable_data<float>();
              auto* output_data = output.mutable_data<float>();
              auto* output_ref_data = output_ref.mutable_data<float>();
              for (int i = 0; i < x_dim.production(); i++) {
                x_data[i] = i;
              }
              for (int i = 0; i < y_dim.production(); i++) {
                y_data[i] = i;
              }
              param.X = &x;
              param.Y = &y;
              param.axis = axis;
              param.Out = &output;
              elementwise_mul.SetParam(param);
              elementwise_mul.Run();
              param.Out = &output_ref;
              elementwise_compute_ref<float>(param, "mul", "");
              for (int i = 0; i < output.dims().production(); i++) {
                EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-5);
              }
            }
          }
        }
      }
    }
  }
}

TEST(fusion_elementwise_mul_activation_arm, retrive_op) {
  auto fusion_elementwise_mul_activation =
      KernelRegistry::Global().Create("fusion_elementwise_mul_activation");
  ASSERT_FALSE(fusion_elementwise_mul_activation.empty());
  ASSERT_TRUE(fusion_elementwise_mul_activation.front());
}

TEST(fusion_elementwise_mul_activation_arm, init) {
  ElementwiseMulActivationCompute<float, PRECISION(kFloat)>
      fusion_elementwise_mul_activation;
  ASSERT_EQ(fusion_elementwise_mul_activation.precision(), PRECISION(kFloat));
  ASSERT_EQ(fusion_elementwise_mul_activation.target(), TARGET(kARM));
}

TEST(fusion_elementwise_mul_activation_arm, compute) {
  ElementwiseMulActivationCompute<float, PRECISION(kFloat)>
      fusion_elementwise_mul_activation;
  operators::FusionElementwiseActivationParam param;
  lite::Tensor x, y, output, output_ref;

#if 1
  for (auto act_type : {"relu"}) {
    for (auto n : {1, 3, 4}) {
      for (auto c : {1, 3, 4}) {
        for (auto h : {1, 3, 4}) {
          for (auto w : {1, 3, 4}) {
            for (auto axis : {-1, 0, 1, 3}) {
              for (auto yd : {std::vector<int64_t>({n}),
                              std::vector<int64_t>({c}),
                              std::vector<int64_t>({h}),
                              std::vector<int64_t>({w}),
                              std::vector<int64_t>({n, c}),
                              std::vector<int64_t>({h, w}),
                              std::vector<int64_t>({n, c, h}),
                              std::vector<int64_t>({n, c, h, w})}) {
#else
  for (auto act_type : {"relu"}) {
    for (auto n : {1, 3, 4, 11}) {
      for (auto c : {1, 3, 4, 11}) {
        for (auto h : {1, 3, 4, 11}) {
          for (auto w : {1, 3, 4, 11}) {
            for (auto axis : {-1, 0, 1, 2, 3}) {
              for (auto yd : {std::vector<int64_t>({n}),
                              std::vector<int64_t>({c}),
                              std::vector<int64_t>({h}),
                              std::vector<int64_t>({w}),
                              std::vector<int64_t>({n, c}),
                              std::vector<int64_t>({c, h}),
                              std::vector<int64_t>({h, w}),
                              std::vector<int64_t>({n, c, h}),
                              std::vector<int64_t>({c, h, w}),
                              std::vector<int64_t>({n, c, h, w})}) {
#endif
                auto x_dim = DDim(std::vector<int64_t>({n, c, h, w}));
                auto y_dim = DDim(yd);
                int axis_t = axis < 0 ? x_dim.size() - y_dim.size() : axis;

                if (axis_t + y_dim.size() > 4) continue;
                bool flag = false;
                for (int i = 0; i < y_dim.size(); i++) {
                  if (x_dim[i + axis_t] != y_dim[i]) flag = true;
                }
                if (flag) continue;

                x.Resize(x_dim);
                y.Resize(y_dim);
                output.Resize(x_dim);
                output_ref.Resize(x_dim);
                auto* x_data = x.mutable_data<float>();
                auto* y_data = y.mutable_data<float>();
                auto* output_data = output.mutable_data<float>();
                auto* output_ref_data = output_ref.mutable_data<float>();
                for (int i = 0; i < x_dim.production(); i++) {
                  float sign = i % 3 == 0 ? -1.0f : 1.0f;
                  x_data[i] = i * sign;
                }
                for (int i = 0; i < y_dim.production(); i++) {
                  float sign = i % 2 == 0 ? 0.5f : -0.5f;
                  y_data[i] = i * sign;
                }
                param.X = &x;
                param.Y = &y;
                param.axis = axis;
                param.Out = &output;
                param.act_type = act_type;
                fusion_elementwise_mul_activation.SetParam(param);
                fusion_elementwise_mul_activation.Run();
                param.Out = &output_ref;
                elementwise_compute_ref<float>(param, "mul", act_type);
                for (int i = 0; i < output.dims().production(); i++) {
                  EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-5);
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(elementwise_max_arm, retrive_op) {
  auto elementwise_max = KernelRegistry::Global().Create("elementwise_max");
  ASSERT_FALSE(elementwise_max.empty());
  ASSERT_TRUE(elementwise_max.front());
}

TEST(elementwise_max_arm, init) {
  ElementwiseMaxCompute elementwise_max;
  ASSERT_EQ(elementwise_max.precision(), PRECISION(kFloat));
  ASSERT_EQ(elementwise_max.target(), TARGET(kARM));
}

TEST(elementwise_max, compute) {
  ElementwiseMaxCompute elementwise_max;
  operators::ElementwiseParam param;
  lite::Tensor x, y, output, output_ref;

#if 1
  for (auto n : {1, 3, 4}) {
    for (auto c : {1, 3, 4}) {
      for (auto h : {1, 3, 4}) {
        for (auto w : {1, 3, 4}) {
          for (auto axis : {-1, 0, 1, 3}) {
            for (auto yd : {std::vector<int64_t>({n}),
                            std::vector<int64_t>({c}),
                            std::vector<int64_t>({h}),
                            std::vector<int64_t>({w}),
                            std::vector<int64_t>({n, c}),
                            std::vector<int64_t>({c, h}),
                            std::vector<int64_t>({c, h, w}),
                            std::vector<int64_t>({n, c, h, w})}) {
#else
  for (auto n : {1, 3, 4, 11}) {
    for (auto c : {1, 3, 4, 11}) {
      for (auto h : {1, 3, 4, 11}) {
        for (auto w : {1, 3, 4, 11}) {
          for (auto axis : {-1, 0, 1, 2, 3}) {
            for (auto yd : {std::vector<int64_t>({n}),
                            std::vector<int64_t>({c}),
                            std::vector<int64_t>({h}),
                            std::vector<int64_t>({w}),
                            std::vector<int64_t>({n, c}),
                            std::vector<int64_t>({c, h}),
                            std::vector<int64_t>({h, w}),
                            std::vector<int64_t>({n, c, h}),
                            std::vector<int64_t>({c, h, w}),
                            std::vector<int64_t>({n, c, h, w})}) {
#endif
              auto x_dim = DDim(std::vector<int64_t>({n, c, h, w}));
              auto y_dim = DDim(yd);
              int axis_t = axis < 0 ? x_dim.size() - y_dim.size() : axis;

              if (axis_t + y_dim.size() > 4) continue;
              bool flag = false;
              for (int i = 0; i < y_dim.size(); i++) {
                if (x_dim[i + axis_t] != y_dim[i]) flag = true;
              }
              if (flag) continue;

              x.Resize(x_dim);
              y.Resize(y_dim);
              output.Resize(x_dim);
              output_ref.Resize(x_dim);
              auto* x_data = x.mutable_data<float>();
              auto* y_data = y.mutable_data<float>();
              auto* output_data = output.mutable_data<float>();
              auto* output_ref_data = output_ref.mutable_data<float>();
              for (int i = 0; i < x_dim.production(); i++) {
                x_data[i] = i;
              }
              for (int i = 0; i < y_dim.production(); i++) {
                y_data[i] = i;
              }
              param.X = &x;
              param.Y = &y;
              param.axis = axis;
              param.Out = &output;
              elementwise_max.SetParam(param);
              elementwise_max.Run();
              param.Out = &output_ref;
              elementwise_compute_ref<float>(param, "max", "");
              for (int i = 0; i < output.dims().production(); i++) {
                EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-5);
              }
            }
          }
        }
      }
    }
  }
}

TEST(fusion_elementwise_max_activation_arm, retrive_op) {
  auto fusion_elementwise_max_activation =
      KernelRegistry::Global().Create("fusion_elementwise_max_activation");
  ASSERT_FALSE(fusion_elementwise_max_activation.empty());
  ASSERT_TRUE(fusion_elementwise_max_activation.front());
}

TEST(fusion_elementwise_max_activation_arm, init) {
  ElementwiseMaxActivationCompute fusion_elementwise_max_activation;
  ASSERT_EQ(fusion_elementwise_max_activation.precision(), PRECISION(kFloat));
  ASSERT_EQ(fusion_elementwise_max_activation.target(), TARGET(kARM));
}

TEST(fusion_elementwise_max_activation_arm, compute) {
  ElementwiseMaxActivationCompute fusion_elementwise_max_activation;
  operators::FusionElementwiseActivationParam param;
  lite::Tensor x, y, output, output_ref;

#if 1
  for (auto act_type : {"relu"}) {
    for (auto n : {1, 3, 4}) {
      for (auto c : {1, 3, 4}) {
        for (auto h : {1, 3, 4}) {
          for (auto w : {1, 3, 4}) {
            for (auto axis : {-1, 0, 1, 3}) {
              for (auto yd : {std::vector<int64_t>({n}),
                              std::vector<int64_t>({c}),
                              std::vector<int64_t>({h}),
                              std::vector<int64_t>({w}),
                              std::vector<int64_t>({n, c}),
                              std::vector<int64_t>({h, w}),
                              std::vector<int64_t>({n, c, h}),
                              std::vector<int64_t>({n, c, h, w})}) {
#else
  for (auto act_type : {"relu"}) {
    for (auto n : {1, 3, 4, 11}) {
      for (auto c : {1, 3, 4, 11}) {
        for (auto h : {1, 3, 4, 11}) {
          for (auto w : {1, 3, 4, 11}) {
            for (auto axis : {-1, 0, 1, 2, 3}) {
              for (auto yd : {std::vector<int64_t>({n}),
                              std::vector<int64_t>({c}),
                              std::vector<int64_t>({h}),
                              std::vector<int64_t>({w}),
                              std::vector<int64_t>({n, c}),
                              std::vector<int64_t>({c, h}),
                              std::vector<int64_t>({h, w}),
                              std::vector<int64_t>({n, c, h}),
                              std::vector<int64_t>({c, h, w}),
                              std::vector<int64_t>({n, c, h, w})}) {
#endif
                auto x_dim = DDim(std::vector<int64_t>({n, c, h, w}));
                auto y_dim = DDim(yd);
                int axis_t = axis < 0 ? x_dim.size() - y_dim.size() : axis;

                if (axis_t + y_dim.size() > 4) continue;
                bool flag = false;
                for (int i = 0; i < y_dim.size(); i++) {
                  if (x_dim[i + axis_t] != y_dim[i]) flag = true;
                }
                if (flag) continue;

                x.Resize(x_dim);
                y.Resize(y_dim);
                output.Resize(x_dim);
                output_ref.Resize(x_dim);
                auto* x_data = x.mutable_data<float>();
                auto* y_data = y.mutable_data<float>();
                auto* output_data = output.mutable_data<float>();
                auto* output_ref_data = output_ref.mutable_data<float>();
                for (int i = 0; i < x_dim.production(); i++) {
                  float sign = i % 3 == 0 ? -1.0f : 1.0f;
                  x_data[i] = i * sign;
                }
                for (int i = 0; i < y_dim.production(); i++) {
                  float sign = i % 2 == 0 ? 0.5f : -0.5f;
                  y_data[i] = (i + 1) * sign;
                }
                param.X = &x;
                param.Y = &y;
                param.axis = axis;
                param.Out = &output;
                param.act_type = act_type;
                fusion_elementwise_max_activation.SetParam(param);
                fusion_elementwise_max_activation.Run();
                param.Out = &output_ref;
                elementwise_compute_ref<float>(param, "max", act_type);
                for (int i = 0; i < output.dims().production(); i++) {
                  EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-5);
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(elementwise_mod_int64_arm, retrive_op) {
  auto elementwise_mod = KernelRegistry::Global().Create("elementwise_mod");
  ASSERT_FALSE(elementwise_mod.empty());
  ASSERT_TRUE(elementwise_mod.front());
}

TEST(elementwise_mod_int64_arm, init) {
  ElementwiseModCompute<int64_t, PRECISION(kInt64)> elementwise_mod;
  ASSERT_EQ(elementwise_mod.precision(), PRECISION(kInt64));
  ASSERT_EQ(elementwise_mod.target(), TARGET(kARM));
}

TEST(elementwise_mod_int64_arm, compute) {
  ElementwiseModCompute<int64_t, PRECISION(kInt64)> elementwise_mod;
  operators::ElementwiseParam param;
  lite::Tensor x, y, output, output_ref;

#if 1
  for (auto n : {1, 3, 4}) {
    for (auto c : {1, 3, 4}) {
      for (auto h : {1, 3, 4}) {
        for (auto w : {1, 3, 4}) {
          for (auto axis : {-1, 0, 1, 3}) {
            for (auto yd : {std::vector<int64_t>({n}),
                            std::vector<int64_t>({c}),
                            std::vector<int64_t>({h}),
                            std::vector<int64_t>({w}),
                            std::vector<int64_t>({n, c}),
                            std::vector<int64_t>({c, h}),
                            std::vector<int64_t>({c, h, w}),
                            std::vector<int64_t>({n, c, h, w})}) {
#else
  for (auto n : {1, 3, 4, 11}) {
    for (auto c : {1, 3, 4, 11}) {
      for (auto h : {1, 3, 4, 11}) {
        for (auto w : {1, 3, 4, 11}) {
          for (auto axis : {-1, 0, 1, 2, 3}) {
            for (auto yd : {std::vector<int64_t>({n}),
                            std::vector<int64_t>({c}),
                            std::vector<int64_t>({h}),
                            std::vector<int64_t>({w}),
                            std::vector<int64_t>({n, c}),
                            std::vector<int64_t>({c, h}),
                            std::vector<int64_t>({h, w}),
                            std::vector<int64_t>({n, c, h}),
                            std::vector<int64_t>({c, h, w}),
                            std::vector<int64_t>({n, c, h, w})}) {
#endif
              auto x_dim = DDim(std::vector<int64_t>({n, c, h, w}));
              auto y_dim = DDim(yd);
              int axis_t = axis < 0 ? x_dim.size() - y_dim.size() : axis;

              if (axis_t + y_dim.size() > 4) continue;
              bool flag = false;
              for (int i = 0; i < y_dim.size(); i++) {
                if (x_dim[i + axis_t] != y_dim[i]) flag = true;
              }
              if (flag) continue;

              x.Resize(x_dim);
              y.Resize(y_dim);
              output.Resize(x_dim);
              output_ref.Resize(x_dim);
              auto* x_data = x.mutable_data<int64_t>();
              auto* y_data = y.mutable_data<int64_t>();
              auto* output_data = output.mutable_data<int64_t>();
              auto* output_ref_data = output_ref.mutable_data<int64_t>();
              for (int i = 0; i < x_dim.production(); i++) {
                x_data[i] = i + 1;
              }
              for (int i = 0; i < y_dim.production(); i++) {
                y_data[i] = y_dim.production() - i;
              }
              param.X = &x;
              param.Y = &y;
              param.axis = axis;
              param.Out = &output;
              elementwise_mod.SetParam(param);
              elementwise_mod.Run();
              param.Out = &output_ref;
              elementwise_imod_compute_ref<int64_t>(param, "");
              for (int i = 0; i < output.dims().production(); i++) {
                if (std::abs(output_data[i] - output_ref_data[i]) > 1e-5 ||
                    std::isnan(output_data[i]) ||
                    std::isnan(output_ref_data[i])) {
                  LOG(FATAL) << "elementwise mod cmp error, i: " << i
                             << ", x_data: " << x_data[i]
                             << ", y_data: " << y_data[i]
                             << ", output_data: " << output_data[i]
                             << ", output_ref_data: " << output_ref_data[i];
                }
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(elementwise_add, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(fusion_elementwise_add_activation, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_mul, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(fusion_elementwise_mul_activation, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_max, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(fusion_elementwise_max_activation, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_mod, kARM, kInt64, kNCHW, def);
