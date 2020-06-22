// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.ddNod
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

#include "lite/kernels/mlu/roi_align_compute.h"

#include <gtest/gtest.h>

#include <memory>
#include <utility>
#include <vector>

#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace mlu {

TEST(roi_align_mlu, retrive_op) {
  auto roi_align =
      KernelRegistry::Global().Create<TARGET(kMLU), PRECISION(kFloat)>(
          "roi_align");
  ASSERT_FALSE(roi_align.empty());
  ASSERT_TRUE(roi_align.front());
}

TEST(roi_align_mlu, init) {
  RoiAlignCompute roi_align;
  ASSERT_EQ(roi_align.precision(), PRECISION(kFloat));
  ASSERT_EQ(roi_align.target(), TARGET(kMLU));
}

TEST(roi_align_mlu, run_test) {
  constexpr int ROI_SIZE = 4;

  // image_height * spatial_scale == featuremap_height, width is also like this
  constexpr int batch_size = 2, channels = 3, featuremap_height = 9,
                featuremap_width = 16, pooled_height = 2, pooled_width = 1,
                num_rois = 3, sampling_rate = 2;
  constexpr float spatial_scale = 0.5;

  lite::Tensor x, rois, out;

  x.Resize(
      lite::DDim({batch_size, channels, featuremap_height, featuremap_width}));
  rois.Resize(lite::DDim({num_rois, ROI_SIZE}));
  // here lod use offset representation: [0, 1), [1, num_rois)
  rois.set_lod({{0, 1, num_rois}});
  out.Resize(lite::DDim({num_rois, channels, pooled_height, pooled_width}));

  auto x_data = x.mutable_data<float>();
  auto rois_data = rois.mutable_data<float>();
  auto out_data = out.mutable_data<float>();

  // {0.0, 1.0, ...}
  std::iota(x_data, x_data + x.dims().production(), 0.0f);
  std::iota(rois_data, rois_data + rois.dims().production(), 0.25f);
  RoiAlignCompute roi_align_op;

  operators::RoiAlignParam param;
  param.X = &x;
  param.ROIs = &rois;
  param.Out = &out;
  param.pooled_height = pooled_height;
  param.pooled_width = pooled_width;
  param.spatial_scale = spatial_scale;
  param.sampling_ratio = sampling_rate;

  // std::unique_ptr<KernelContext> ctx(new KernelContext);
  // ctx->As<MLUContext>();
  // roi_align_op.SetContext(std::move(ctx));

  CNRT_CALL(cnrtInit(0));
  // cnrtInvokeFuncParam_t forward_param;
  // u32_t affinity = 1;
  // int data_param = 1;
  // forward_param.data_parallelism = &data_param;
  // forward_param.affinity = &affinity;
  // forward_param.end = CNRT_PARAM_END;
  cnrtDev_t dev_handle;
  CNRT_CALL(cnrtGetDeviceHandle(&dev_handle, 0));
  CNRT_CALL(cnrtSetCurrentDevice(dev_handle));
  cnrtQueue_t queue;
  CNRT_CALL(cnrtCreateQueue(&queue));

  roi_align_op.SetParam(param);
  roi_align_op.Run(queue);

  CNRT_CALL(cnrtDestroyQueue(queue));

  std::vector<float> ref_results = {14.625,
                                    22.625,
                                    158.625,
                                    166.625,
                                    302.625,
                                    310.625,

                                    480.625,
                                    488.625,
                                    624.625,
                                    632.625,
                                    768.625,
                                    776.625,

                                    514.625,
                                    522.625,
                                    658.625,
                                    666.625,
                                    802.625,
                                    810.625};
  for (int i = 0; i < out.dims().production(); i++) {
    EXPECT_NEAR(out_data[i], ref_results[i], (4e-3f * ref_results[i]));
  }
}

}  // namespace mlu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(roi_align, kMLU, kFloat, kNCHW, def);
