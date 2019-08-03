/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "lite/api/android/jni/native/paddle_init_jni.h"

#include <memory>

#include "lite/api/paddle_lite_factory_helper.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#ifndef LITE_ON_TINY_PUBLISH
#include "lite/api/paddle_use_passes.h"
#endif
#include "lite/kernels/arm/activation_compute.h"
#include "lite/kernels/arm/argmax_compute.h"
#include "lite/kernels/arm/axpy_compute.h"
#include "lite/kernels/arm/batch_norm_compute.h"
#include "lite/kernels/arm/beam_search_compute.h"
#include "lite/kernels/arm/beam_search_decode_compute.h"
#include "lite/kernels/arm/box_coder_compute.h"
#include "lite/kernels/arm/calib_compute.h"
// #include "lite/kernels/arm/compare_compute.h"
#include "lite/kernels/arm/concat_compute.h"
#include "lite/kernels/arm/conv_compute.h"
#include "lite/kernels/arm/conv_transpose_compute.h"
#include "lite/kernels/arm/crop_compute.h"
#include "lite/kernels/arm/decode_bboxes_compute.h"
#include "lite/kernels/arm/density_prior_box_compute.h"
#include "lite/kernels/arm/dropout_compute.h"
#include "lite/kernels/arm/elementwise_compute.h"
#include "lite/kernels/arm/fc_compute.h"
#include "lite/kernels/arm/gru_compute.h"
#include "lite/kernels/arm/gru_unit_compute.h"
#include "lite/kernels/arm/im2sequence_compute.h"
#include "lite/kernels/arm/increment_compute.h"
// #include "lite/kernels/arm/logical_compute.h"
#include "lite/kernels/arm/lookup_table_compute.h"
#include "lite/kernels/arm/lrn_compute.h"
#include "lite/kernels/arm/mul_compute.h"
#include "lite/kernels/arm/multiclass_nms_compute.h"
#include "lite/kernels/arm/negative_compute.h"
#include "lite/kernels/arm/norm_compute.h"
#include "lite/kernels/arm/pad2d_compute.h"
#include "lite/kernels/arm/pool_compute.h"
#include "lite/kernels/arm/prior_box_compute.h"
#include "lite/kernels/arm/read_from_array_compute.h"
#include "lite/kernels/arm/reduce_max_compute.h"
#include "lite/kernels/arm/scale_compute.h"
#include "lite/kernels/arm/sequence_expand_compute.h"
#include "lite/kernels/arm/sequence_pool_compute.h"
#include "lite/kernels/arm/sequence_softmax_compute.h"
#include "lite/kernels/arm/softmax_compute.h"
#include "lite/kernels/arm/split_compute.h"
#include "lite/kernels/arm/topk_compute.h"
#include "lite/kernels/arm/transpose_compute.h"
#include "lite/kernels/arm/write_to_array_compute.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ARM_KERNEL_POINTER(kernel_class_name__)                    \
  std::unique_ptr<paddle::lite::kernels::arm::kernel_class_name__> \
      p##kernel_class_name__(                                      \
          new paddle::lite::kernels::arm::kernel_class_name__);

namespace paddle {
namespace lite_api {

/**
 * Not sure why, we have to initial a pointer first for kernels.
 * Otherwise it throws null pointer error when do KernelRegistor.
 */
static void use_arm_kernels() {
  ARM_KERNEL_POINTER(ReluCompute);
  ARM_KERNEL_POINTER(LeakyReluCompute);
  ARM_KERNEL_POINTER(ReluClippedCompute);
  ARM_KERNEL_POINTER(PReluCompute);
  ARM_KERNEL_POINTER(SigmoidCompute);
  ARM_KERNEL_POINTER(TanhCompute);
  ARM_KERNEL_POINTER(SwishCompute);
  ARM_KERNEL_POINTER(ArgmaxCompute);
  ARM_KERNEL_POINTER(AxpyCompute);
  ARM_KERNEL_POINTER(BatchNormCompute);
  ARM_KERNEL_POINTER(BeamSearchCompute);
  ARM_KERNEL_POINTER(BeamSearchDecodeCompute);
  ARM_KERNEL_POINTER(BoxCoderCompute);
  ARM_KERNEL_POINTER(CalibComputeFp32ToInt8);
  ARM_KERNEL_POINTER(CalibComputeInt8ToFp32);
  // ARM_KERNEL_POINTER(CompareCompute<paddle::lite::kernels::arm::_LessThanFunctor>);
  // ARM_KERNEL_POINTER(CompareCompute<paddle::lite::kernels::arm::_EqualFunctor>);
  // ARM_KERNEL_POINTER(CompareCompute<paddle::lite::kernels::arm::_NotEqualFunctor>);
  // ARM_KERNEL_POINTER(CompareCompute<paddle::lite::kernels::arm::_LessEqualFunctor>);
  // ARM_KERNEL_POINTER(CompareCompute<paddle::lite::kernels::arm::_GreaterThanFunctor>);
  // ARM_KERNEL_POINTER(CompareCompute<paddle::lite::kernels::arm::_GreaterEqualFunctor>);
  ARM_KERNEL_POINTER(ConcatCompute);
  ARM_KERNEL_POINTER(ConvCompute);
  std::unique_ptr<paddle::lite::kernels::arm::ConvComputeInt8<PRECISION(kInt8)>>
      pConvComputeInt8Int8(
          new paddle::lite::kernels::arm::ConvComputeInt8<PRECISION(kInt8)>);
  std::unique_ptr<
      paddle::lite::kernels::arm::ConvComputeInt8<PRECISION(kFloat)>>
      pConvComputeInt8FLoat(
          new paddle::lite::kernels::arm::ConvComputeInt8<PRECISION(kFloat)>);
  ARM_KERNEL_POINTER(Conv2DTransposeCompute);
  ARM_KERNEL_POINTER(CropCompute);
  ARM_KERNEL_POINTER(DecodeBboxesCompute);
  ARM_KERNEL_POINTER(DensityPriorBoxCompute);
  ARM_KERNEL_POINTER(DropoutCompute);
  ARM_KERNEL_POINTER(ElementwiseAddCompute);
  ARM_KERNEL_POINTER(ElementwiseAddActivationCompute);
  ARM_KERNEL_POINTER(ElementwiseMulCompute);
  ARM_KERNEL_POINTER(ElementwiseMulActivationCompute);
  ARM_KERNEL_POINTER(ElementwiseMaxCompute);
  ARM_KERNEL_POINTER(ElementwiseMaxActivationCompute);
  ARM_KERNEL_POINTER(FcCompute);
  std::unique_ptr<paddle::lite::kernels::arm::FcComputeInt8<PRECISION(kInt8)>>
      pFcComputeInt8Int8(
          new paddle::lite::kernels::arm::FcComputeInt8<PRECISION(kInt8)>);
  std::unique_ptr<paddle::lite::kernels::arm::FcComputeInt8<PRECISION(kFloat)>>
      pFcComputeInt8FLoat(
          new paddle::lite::kernels::arm::FcComputeInt8<PRECISION(kFloat)>);
  ARM_KERNEL_POINTER(GRUCompute);
  ARM_KERNEL_POINTER(GRUUnitCompute);
  ARM_KERNEL_POINTER(Im2SequenceCompute);
  ARM_KERNEL_POINTER(IncrementCompute);
  // ARM_KERNEL_POINTER(BinaryLogicalCompute<paddle::lite::kernels::arm::_LogicalXorFunctor>);
  // ARM_KERNEL_POINTER(BinaryLogicalCompute<paddle::lite::kernels::arm::_LogicalAndFunctor>);
  // ARM_KERNEL_POINTER(BinaryLogicalCompute<paddle::lite::kernels::arm::_LogicalOrFunctor>);
  // ARM_KERNEL_POINTER(BinaryLogicalCompute<paddle::lite::kernels::arm::__LogicalNotFunctor>);
  ARM_KERNEL_POINTER(LookupTableCompute);
  ARM_KERNEL_POINTER(LrnCompute);
  ARM_KERNEL_POINTER(MulCompute);
  ARM_KERNEL_POINTER(MulticlassNmsCompute);
  ARM_KERNEL_POINTER(NegativeCompute);
  ARM_KERNEL_POINTER(NormCompute);
  ARM_KERNEL_POINTER(Pad2dCompute);
  ARM_KERNEL_POINTER(PoolCompute);
  ARM_KERNEL_POINTER(PriorBoxCompute);
  ARM_KERNEL_POINTER(ReadFromArrayCompute);
  ARM_KERNEL_POINTER(ReduceMaxCompute);
  ARM_KERNEL_POINTER(ScaleCompute);
  ARM_KERNEL_POINTER(SequenceExpandCompute);
  ARM_KERNEL_POINTER(SequencePoolCompute);
  ARM_KERNEL_POINTER(SequenceSoftmaxCompute);
  ARM_KERNEL_POINTER(SoftmaxCompute);
  ARM_KERNEL_POINTER(SplitCompute);
  ARM_KERNEL_POINTER(TopkCompute);
  ARM_KERNEL_POINTER(TransposeCompute);
  ARM_KERNEL_POINTER(Transpose2Compute);
  ARM_KERNEL_POINTER(WriteToArrayCompute);
}

JNIEXPORT void JNICALL
Java_com_baidu_paddle_lite_PaddleLiteInitializer_initNative(JNIEnv *env,
                                                            jclass jclazz) {
  use_arm_kernels();
}

}  // namespace lite_api
}  // namespace paddle

#ifdef __cplusplus
}
#endif
