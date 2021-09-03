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
#include <memory>
#include <random>
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/opencl/test_helper.h"

#define FP16_MAX_DIFF (5e-1)
namespace paddle {
namespace lite {
void box_coder_ref(float* proposals_data,
                   const float* anchors_data,
                   const float* bbox_deltas_data,
                   const float* variances_data,
                   int axis,
                   bool box_normalized,
                   std::string code_type,
                   int row,
                   int col) {
  if (code_type == "decode_center_size") {
    int anchor_len = 4;
    int out_len = 4;
    int var_len = 4;
    int delta_len = 4;
    float normalized = !box_normalized ? 1.f : 0;

    for (int64_t row_id = 0; row_id < row; ++row_id) {
      for (int64_t col_id = 0; col_id < col; ++col_id) {
        size_t delta_offset = row_id * col * delta_len + col_id * delta_len;
        size_t out_offset = row_id * col * out_len + col_id * out_len;
        int prior_box_offset =
            axis == 0 ? col_id * anchor_len : row_id * anchor_len;
        int var_offset = axis == 0 ? col_id * var_len : row_id * var_len;
        auto anchor_data_tmp = anchors_data + prior_box_offset;
        auto bbox_deltas_data_tmp = bbox_deltas_data + delta_offset;
        auto proposals_data_tmp = proposals_data + out_offset;
        auto anchor_width =
            anchor_data_tmp[2] - anchor_data_tmp[0] + normalized;
        auto anchor_height =
            anchor_data_tmp[3] - anchor_data_tmp[1] + normalized;
        auto anchor_center_x = anchor_data_tmp[0] + 0.5 * anchor_width;
        auto anchor_center_y = anchor_data_tmp[1] + 0.5 * anchor_height;
        float bbox_center_x = 0, bbox_center_y = 0;
        float bbox_width = 0, bbox_height = 0;

        auto variances_data_tmp = variances_data + var_offset;
        bbox_center_x =
            variances_data_tmp[0] * bbox_deltas_data_tmp[0] * anchor_width +
            anchor_center_x;
        bbox_center_y =
            variances_data_tmp[1] * bbox_deltas_data_tmp[1] * anchor_height +
            anchor_center_y;
        bbox_width = std::exp(variances_data_tmp[2] * bbox_deltas_data_tmp[2]) *
                     anchor_width;
        bbox_height =
            std::exp(variances_data_tmp[3] * bbox_deltas_data_tmp[3]) *
            anchor_height;
        proposals_data_tmp[0] = bbox_center_x - bbox_width / 2;
        proposals_data_tmp[1] = bbox_center_y - bbox_height / 2;
        proposals_data_tmp[2] = bbox_center_x + bbox_width / 2 - normalized;
        proposals_data_tmp[3] = bbox_center_y + bbox_height / 2 - normalized;
      }
    }
  } else if (code_type == "encode_center_size") {
    LOG(FATAL) << "not implemented type: " << code_type;
  } else {
    LOG(FATAL) << "not supported type: " << code_type;
  }
}
// #define BOXCODER_FP16_LOOP_TEST
// #define BOXCODER_FP16_PRINT_RESULT
TEST(box_coder_image2d, compute) {
#ifdef BOXCODER_FP16_LOOP_TEST
  for (auto n : {1, 2, 3, 4}) {
    for (auto m : {1, 3, 4, 8}) {
      for (auto norm : {true}) {
        for (auto code_type : {"decode_center_size"}) {
          for (auto axis : {0}) {
#else
  const int n = 1;
  const int m = 1;
  const bool norm = true;
  const std::string code_type = "decode_center_size";
  const int axis = 0;
#endif  // BOXCODER_FP16_LOOP_TEST

            LOG(INFO) << "======== input shape[n,c,h,w]:" << n << " " << m
                      << " ========";
            LOG(INFO) << "======== parameters: norm = " << norm
                      << ", axis = " << axis << "code_type: " << code_type;

            auto kernels =
                KernelRegistry::Global().Create("box_coder",
                                                TARGET(kOpenCL),
                                                PRECISION(kFP16),
                                                DATALAYOUT(kImageDefault));
            ASSERT_FALSE(kernels.empty());
            auto kernel = std::move(kernels.front());
            LOG(INFO) << "get kernel:" << kernel->doc();

            lite::Tensor prior_box, prior_box_var, target_box, output_box;
            operators::BoxCoderParam param;
            param.prior_box = &prior_box;
            param.prior_box_var = &prior_box_var;
            param.target_box = &target_box;
            param.proposals = &output_box;
            param.axis = axis;
            param.box_normalized = norm;
            param.code_type = code_type;

            std::unique_ptr<KernelContext> context(new KernelContext);
            context->As<OpenCLContext>().InitOnce();

            kernel->SetParam(param);
            std::unique_ptr<KernelContext> boxcoder_context(new KernelContext);
            context->As<OpenCLContext>().CopySharedTo(
                &(boxcoder_context->As<OpenCLContext>()));
            kernel->SetContext(std::move(boxcoder_context));

            const DDim prior_box_dims =
                DDim(std::vector<DDim::value_type>{1, 1, m, 4});
            const DDim prior_box_var_dims =
                DDim(std::vector<DDim::value_type>{1, 1, m, 4});
            const DDim target_box_dims =
                DDim(std::vector<DDim::value_type>{1, n, m, 4});
            const DDim out_dim =
                DDim(std::vector<DDim::value_type>{1, n, m, 4});
            prior_box.Resize(prior_box_dims);
            prior_box_var.Resize(prior_box_var_dims);
            target_box.Resize(target_box_dims);
            output_box.Resize(out_dim);

            std::vector<float> prior_box_data(prior_box_dims.production());
            std::vector<float> prior_box_var_data(
                prior_box_var_dims.production());
            std::vector<float> target_box_data(target_box_dims.production());
            for (int i = 0; i < prior_box_dims.production(); i++) {
              prior_box_data[i] = i * 1.1 / prior_box_dims.production();
            }
            for (int i = 0; i < prior_box_var_dims.production(); i++) {
              prior_box_var_data[i] = i * 1.2 / prior_box_var_dims.production();
            }
            for (int i = 0; i < target_box_dims.production(); i++) {
              target_box_data[i] = i * 1.3 / target_box_dims.production();
            }

            LOG(INFO) << "prepare input";
            CLImageConverterDefault* default_converter =
                new CLImageConverterDefault();
            DDim prior_box_image_shape =
                default_converter->InitImageDimInfoWith(prior_box_dims);
            LOG(INFO) << "prior_box_image_shape = " << prior_box_image_shape[0]
                      << " " << prior_box_image_shape[1];
            std::vector<half_t> prior_box_image_data(
                prior_box_image_shape.production() * 4);  // 4 : RGBA
            default_converter->NCHWToImage(prior_box_data.data(),
                                           prior_box_image_data.data(),
                                           prior_box_dims);
            auto* prior_box_image = prior_box.mutable_data<half_t, cl::Image2D>(
                prior_box_image_shape[0],
                prior_box_image_shape[1],
                prior_box_image_data.data());

            DDim prior_box_var_image_shape =
                default_converter->InitImageDimInfoWith(prior_box_var_dims);
            LOG(INFO) << "prior_box_var_image_shape = "
                      << prior_box_var_image_shape[0] << " "
                      << prior_box_var_image_shape[1];
            std::vector<half_t> prior_box_var_image_data(
                prior_box_var_image_shape.production() * 4);  // 4 : RGBA
            default_converter->NCHWToImage(prior_box_var_data.data(),
                                           prior_box_var_image_data.data(),
                                           prior_box_var_dims);
            auto* prior_box_var_image =
                prior_box_var.mutable_data<half_t, cl::Image2D>(
                    prior_box_var_image_shape[0],
                    prior_box_var_image_shape[1],
                    prior_box_var_image_data.data());

            DDim target_box_image_shape =
                default_converter->InitImageDimInfoWith(target_box_dims);
            LOG(INFO) << "target_box_image_shape = "
                      << target_box_image_shape[0] << " "
                      << target_box_image_shape[1];
            std::vector<half_t> target_box_image_data(
                target_box_image_shape.production() * 4);  // 4 : RGBA
            default_converter->NCHWToImage(target_box_data.data(),
                                           target_box_image_data.data(),
                                           target_box_dims);
            auto* target_box_image =
                target_box.mutable_data<half_t, cl::Image2D>(
                    target_box_image_shape[0],
                    target_box_image_shape[1],
                    target_box_image_data.data());

            DDim out_image_shape =
                default_converter->InitImageDimInfoWith(out_dim);
            LOG(INFO) << "out_image_shape = " << out_image_shape[0] << " "
                      << out_image_shape[1];
            auto* out_image = output_box.mutable_data<half_t, cl::Image2D>(
                out_image_shape[0], out_image_shape[1]);
            kernel->Launch();

            CLRuntime::Global()->command_queue().finish();

            lite::Tensor out_ref_tensor;
            out_ref_tensor.Resize(out_dim);
            box_coder_ref(out_ref_tensor.mutable_data<float>(),
                          prior_box_data.data(),
                          target_box_data.data(),
                          prior_box_var_data.data(),
                          axis,
                          norm,
                          code_type,
                          target_box_dims[0],
                          target_box_dims[1]);

            const size_t cl_image2d_row_pitch{0};
            const size_t cl_image2d_slice_pitch{0};
            half_t* out_image_data =
                new half_t[40000];  // [out_image_shape.production() * 4];
            TargetWrapperCL::ImgcpySync(out_image_data,
                                        out_image,
                                        out_image_shape[0],
                                        out_image_shape[1],
                                        cl_image2d_row_pitch,
                                        cl_image2d_slice_pitch,
                                        IoDirection::DtoH);
            float* out_data = new float[out_image_shape.production() * 4];
            default_converter->ImageToNCHW(
                out_image_data, out_data, out_image_shape, out_dim);
// result
#ifdef BOXCODER_FP16_PRINT_RESULT
            LOG(INFO) << "---- print kernel result (input -> output) ----";
            for (int eidx = 0; eidx < out_dim.production(); ++eidx) {
              std::cout << target_box_data[eidx] << " -> " << out_data[eidx]
                        << std::endl;
            }
#endif  // BOXCODER_FP16_PRINT_RESULT
            const float* out_ref = out_ref_tensor.data<float>();
            for (int i = 0; i < out_dim.production(); i++) {
              auto abs_diff = abs(out_data[i] - out_ref[i]);
              auto relative_diff =
                  COMPUTE_RELATIVE_DIFF(out_data[i], out_ref[i]);
              EXPECT_EQ((relative_diff <= FP16_MAX_DIFF) ||
                            (abs_diff <= FP16_MAX_DIFF),
                        true);
              if ((relative_diff > FP16_MAX_DIFF) &&
                  (abs_diff > FP16_MAX_DIFF)) {
                LOG(ERROR) << "error idx:" << i << ", in_data[" << i
                           << "]: " << target_box_data[i] << ", out_data[" << i
                           << "]: " << out_data[i] << ", out_ref[" << i
                           << "]: " << out_ref[i] << ", abs_diff: " << abs_diff
                           << ", relative_diff: " << relative_diff
                           << ", FP16_MAX_DIFF: " << FP16_MAX_DIFF;
              }
            }
#ifdef BOXCODER_FP16_LOOP_TEST
          }  // axis
        }    // code_type
      }      // norm
    }        // m
  }          // n
#else
// nothing to do.
#endif
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(box_coder, kOpenCL, kFP16, kImageDefault, ImageDefault);
