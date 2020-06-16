// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <algorithm>
#include <map>
#include <memory>
#include <vector>
#include "lite/backends/mlu/mlu_utils.h"
#include "lite/core/kernel.h"
#include "lite/kernels/mlu/bridges/tensor.h"
#include "lite/kernels/mlu/bridges/utility.h"
#include "lite/kernels/mlu/mlu_operator.h"
#include "lite/operators/cast_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace mlu {

template <lite_api::PrecisionType in_dtype, lite_api::PrecisionType out_dtype>
class CastCompute
    : public KernelLite<TARGET(kMLU), PRECISION(kFloat), DATALAYOUT(kNHWC)> {
 public:
  using param_t = operators::CastParam;

  void Run() override {
    auto param = param_.get_mutable<param_t>();
    auto& mlu_context = this->ctx_->template As<MLUContext>();
    auto in_dims = param->X->dims().Vectorize();
    // key to map op
    std::vector<int> ishape;
    std::transform(in_dims.cbegin(),
                   in_dims.cend(),
                   std::back_inserter(ishape),
                   [](DDim::value_type in) { return static_cast<int>(in); });

    // find compiled instruction at ishape
    auto op_iter = inst_map_.find(ishape);
    if (op_iter == inst_map_.end()) {
      auto res = inst_map_.insert(
          {ishape, CompileOperator(param, &mlu_context, ishape)});
      CHECK(res.second);
      op_iter = res.first;
    }

    // prepare param
    auto exec_queue = mlu_context.exec_queue();
    cnrtInvokeFuncParam_t forward_param = mlu_context.forward_param();
    int data_param = 1;
    forward_param.data_parallelism = &data_param;
    u32_t affinity = mlu_context.affinity();
    forward_param.affinity = &affinity;
    forward_param.end = CNRT_PARAM_END;

    // get input and output
    param->Out->set_precision(out_dtype);
    const void* input = param->X->template data<
        typename subgraph::mlu::MLUTypeTraits<in_dtype>::type>();
    /* void* output = param->Out->mutable_data(TARGET(kMLU), out_size); */
    void* output = param->Out->template mutable_data<
        typename subgraph::mlu::MLUTypeTraits<out_dtype>::type>(TARGET(kMLU));

    // compute op
    CNML_CALL(cnmlComputeCastOpForward_V3(op_iter->second->cnml_op,
                                          const_cast<void*>(input),
                                          output,
                                          &forward_param,
                                          exec_queue));
  }

  ~CastCompute() override{};

 private:
  inline cnmlCastType_t GetCastType(param_t* param) {
    CHECK_EQ(subgraph::mlu::MLUTypeTraits<in_dtype>::proto_type,
             param->in_dtype);
    CHECK_EQ(subgraph::mlu::MLUTypeTraits<out_dtype>::proto_type,
             param->out_dtype);

    if (in_dtype == PRECISION(kFP16) && out_dtype == PRECISION(kFloat)) {
      VLOG(4) << "choose float16 to float32";
      return CNML_CAST_FLOAT16_TO_FLOAT32;
    } else if (in_dtype == PRECISION(kFloat) && out_dtype == PRECISION(kFP16)) {
      VLOG(4) << "choose float32 to float16";
      return CNML_CAST_FLOAT32_TO_FLOAT16;
    } else {
      CHECK(0) << "Unsupported cast type";
    }
    return CNML_CAST_FLOAT32_TO_FLOAT16;
  }

  std::shared_ptr<MLUOperator> CompileOperator(param_t* param,
                                               MLUContext* ctx,
                                               std::vector<int> dims) {
    VLOG(4) << "compile cast operator";
    // get cast type
    auto cast_type = GetCastType(param);

    // prepare op and io tensor
    auto op = std::make_shared<MLUOperator>();
    op->input_tensors.emplace_back();
    op->output_tensors.emplace_back();

    int* dim_strides = nullptr;
    CNML_CALL(cnmlCreateTensor_V2(&op->input_tensors[0], CNML_TENSOR));
    CNML_CALL(cnmlSetTensorShape_V2(
        op->input_tensors[0], dims.size(), dims.data(), dim_strides));
    CNML_CALL(cnmlSetTensorDataType(
        op->input_tensors[0],
        subgraph::mlu::MLUTypeTraits<in_dtype>::cnml_type));

    CNML_CALL(cnmlCreateTensor_V2(&op->output_tensors[0], CNML_TENSOR));
    CNML_CALL(cnmlSetTensorShape_V2(
        op->output_tensors[0], dims.size(), dims.data(), dim_strides));
    CNML_CALL(cnmlSetTensorDataType(
        op->output_tensors[0],
        subgraph::mlu::MLUTypeTraits<out_dtype>::cnml_type));

    CNML_CALL(cnmlCreateCastOp(
        &op->cnml_op, cast_type, op->input_tensors[0], op->output_tensors[0]));
    CNML_CALL(cnmlSetBaseOpCorenum(op->cnml_op, ctx->MLUCoreNumber()));
    CNML_CALL(cnmlSetBaseOpCoreVersion(op->cnml_op, ctx->MLUCoreVersion()));
    CNML_CALL(cnmlCompileBaseOp_V2(op->cnml_op));
    return op;
  }

 private:
  std::map<std::vector<int>, std::shared_ptr<MLUOperator>> inst_map_;
};

using CastFp32toFp16 =
    paddle::lite::kernels::mlu::CastCompute<PRECISION(kFloat),
                                            PRECISION(kFP16)>;
using CastFp16toFp32 =
    paddle::lite::kernels::mlu::CastCompute<PRECISION(kFP16),
                                            PRECISION(kFloat)>;

}  // namespace mlu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
