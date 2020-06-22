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

namespace paddle {
namespace lite {
namespace kernels {
namespace mlu {

void RoiAlignCompute::Run() {
  auto& mlu_context = this->ctx_->template As<MLUContext>();
  auto& exec_queue = mlu_context.exec_queue();
  this->Run(exec_queue);
}

void RoiAlignCompute::Run(const cnrtQueue_t& exec_queue) {
  auto& param = this->Param<param_t>();

  auto* rois = param.ROIs;
  auto rois_dims = rois->dims();
  int rois_num = rois_dims[0];
  if (rois_num == 0) {
    return;
  }

  auto* in = param.X;
  auto* out = param.Out;
  float spatial_scale = param.spatial_scale;
  int pooled_height = param.pooled_height;
  int pooled_width = param.pooled_width;
  int sampling_ratio = param.sampling_ratio;

  half spatial_scale_half;
  cnrtConvertFloatToHalf(&spatial_scale_half, spatial_scale);

  auto in_dims = in->dims();
  // int batch_size = in_dims[0];
  int channels = in_dims[1];
  int height = in_dims[2];
  int width = in_dims[3];
  auto out_dims = out->dims();

  std::vector<int> roi_ind_vec(rois_num);
  auto rois_lod = rois->lod().back();
  for (int n = 0, rois_batch_size = rois_lod.size() - 1; n < rois_batch_size;
       ++n) {
    for (size_t i = rois_lod[n]; i < rois_lod[n + 1]; ++i) {
      roi_ind_vec[i] = n;
    }
  }

  auto* input_data = in->data<float>();
  auto* output_data = out->mutable_data<float>();
  auto* rois_data = rois->data<float>();

  std::vector<half> input_tmp_vec(in_dims.production());
  std::vector<half> rois_tmp_vec(rois_dims.production());
  std::vector<half> output_tmp_vec(out_dims.production());

  std::vector<int> nchw2nhwc_dimorder{0, 2, 3, 1};
  std::vector<int> tmp_in_dims;
  for (int i = 0; i < in_dims.size(); i++) {
    tmp_in_dims.emplace_back(static_cast<int>(in_dims[i]));
  }
  cnrtTransOrderAndCast(const_cast<float*>(input_data),
                        CNRT_FLOAT32,
                        input_tmp_vec.data(),
                        CNRT_FLOAT16,
                        NULL,
                        tmp_in_dims.size(),
                        tmp_in_dims.data(),
                        nchw2nhwc_dimorder.data());
  cnrtCastDataType(const_cast<float*>(rois_data),
                   CNRT_FLOAT32,
                   const_cast<half*>(rois_tmp_vec.data()),
                   CNRT_FLOAT16,
                   rois_dims.production(),
                   NULL);

  void *input_mlu_data = nullptr, *rois_mlu_data = nullptr,
       *roi_batch_id_mlu_data = nullptr, *output_mlu_data = nullptr;
  cnrtMalloc(&input_mlu_data,
             input_tmp_vec.size() * sizeof(input_tmp_vec.front()));
  cnrtMemcpy(input_mlu_data,
             input_tmp_vec.data(),
             input_tmp_vec.size() * sizeof(input_tmp_vec.front()),
             CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMalloc(&rois_mlu_data,
             rois_tmp_vec.size() * sizeof(rois_tmp_vec.front()));
  cnrtMemcpy(rois_mlu_data,
             rois_tmp_vec.data(),
             rois_tmp_vec.size() * sizeof(rois_tmp_vec.front()),
             CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMalloc(&roi_batch_id_mlu_data,
             roi_ind_vec.size() * sizeof(roi_ind_vec.front()));
  cnrtMemcpy(roi_batch_id_mlu_data,
             roi_ind_vec.data(),
             roi_ind_vec.size() * sizeof(roi_ind_vec.front()),
             CNRT_MEM_TRANS_DIR_HOST2DEV);

  // malloc output memory on device
  cnrtMalloc(&output_mlu_data,
             output_tmp_vec.size() * sizeof(output_tmp_vec.front()));

  // prepare kernel params
  cnrtKernelParamsBuffer_t params;
  cnrtGetKernelParamsBuffer(&params);
  cnrtKernelParamsBufferAddParam(
      params, &input_mlu_data, sizeof(input_mlu_data));
  cnrtKernelParamsBufferAddParam(params, &rois_mlu_data, sizeof(rois_mlu_data));
  cnrtKernelParamsBufferAddParam(
      params, &roi_batch_id_mlu_data, sizeof(roi_batch_id_mlu_data));
  cnrtKernelParamsBufferAddParam(
      params, &output_mlu_data, sizeof(output_mlu_data));
  cnrtKernelParamsBufferAddParam(params, &height, sizeof(height));
  cnrtKernelParamsBufferAddParam(params, &width, sizeof(width));
  cnrtKernelParamsBufferAddParam(params, &channels, sizeof(channels));
  cnrtKernelParamsBufferAddParam(params, &pooled_height, sizeof(pooled_height));
  cnrtKernelParamsBufferAddParam(params, &pooled_width, sizeof(pooled_width));
  cnrtKernelParamsBufferAddParam(params, &rois_num, sizeof(rois_num));
  cnrtKernelParamsBufferAddParam(
      params, &spatial_scale_half, sizeof(spatial_scale_half));
  cnrtKernelParamsBufferAddParam(
      params, &sampling_ratio, sizeof(sampling_ratio));

  cnrtDim3_t task_dims;
  task_dims.x = 1, task_dims.y = 1, task_dims.z = 1;
  cnrtFunctionType_t func_type = CNRT_FUNC_TYPE_BLOCK;

  // invoke kernel and sync to compute on MLU
  CNRT_CALL(cnrtInvokeKernel_V2(reinterpret_cast<void*>(&roi_align_kernel),
                                task_dims,
                                params,
                                func_type,
                                exec_queue));
  CNRT_CALL(cnrtSyncQueue(exec_queue));

  cnrtMemcpy(output_tmp_vec.data(),
             output_mlu_data,
             output_tmp_vec.size() * sizeof(output_tmp_vec.front()),
             CNRT_MEM_TRANS_DIR_DEV2HOST);
  std::vector<int> tmp_out_dims;
  for (int i = 0; i < out_dims.size(); i++) {
    // out_dims = {N, C, H, W}, tmp_out_dims = {N, H, W, C}
    tmp_out_dims.emplace_back(out_dims[nchw2nhwc_dimorder[i]]);
  }
  std::vector<int> nhwc2nchw_dimorder{0, 3, 1, 2};
  cnrtTransOrderAndCast(output_tmp_vec.data(),
                        CNRT_FLOAT16,
                        output_data,
                        CNRT_FLOAT32,
                        NULL,
                        tmp_out_dims.size(),
                        tmp_out_dims.data(),
                        nhwc2nchw_dimorder.data());

  // realease resource
  cnrtDestroyKernelParamsBuffer(params);
  cnrtFree(input_mlu_data);
  cnrtFree(rois_mlu_data);
  cnrtFree(roi_batch_id_mlu_data);
  cnrtFree(output_mlu_data);
}

}  // namespace mlu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(roi_align,
                     kMLU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::mlu::RoiAlignCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindInput("ROIs",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
