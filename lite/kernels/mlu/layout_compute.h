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

#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "lite/backends/x86/math/math_function.h"
#include "lite/core/kernel.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#include "lite/kernels/mlu/bridges/utility.h"
#include "lite/kernels/mlu/mlu_operator.h"
#include "lite/operators/layout_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace mlu {

template <lite::TargetType Target, typename T>
inline void LayoutTransCompute(const int dim,
                               const lite::Context<Target>& context,
                               const lite::Tensor& in,
                               lite::Tensor* out,
                               const std::vector<int>& axis) {
  switch (dim) {
    case 2:
      paddle::lite::x86::math::Transpose<lite::TargetType::kX86, T, 2> trans2;
      trans2(context, in, out, axis);
      break;
    case 3:
      paddle::lite::x86::math::Transpose<lite::TargetType::kX86, T, 3> trans3;
      trans3(context, in, out, axis);
      break;
    case 4:
      paddle::lite::x86::math::Transpose<lite::TargetType::kX86, T, 4> trans4;
      trans4(context, in, out, axis);
      break;
    default:
      CHECK(0) << ("Unsupport dim in mlu layout");
  }
}

template <PrecisionType Precision>
class LayoutNchwToNhwcCompute
    : public KernelLite<TARGET(kX86), Precision, DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::LayoutParam;

  void Run() override {
    auto& param = this->template Param<param_t>();
    auto* x = param.x;
    auto* out = param.y;
    out->template mutable_data<
        typename subgraph::mlu::MLUTypeTraits<Precision>::type>();
    auto x_ndims = param.x->dims().size();
    auto& context = this->ctx_->template As<X86Context>();

    const auto origin_dims = out->dims().Vectorize();

    std::vector<int> axis;
    switch (x_ndims) {
      case 2:
        axis = {0, 1};
        break;
      case 3:
        axis = {0, 2, 1};
        out->Resize(std::vector<int64_t>{
            origin_dims[0], origin_dims[2], origin_dims[1]});
        break;
      case 4:
        axis = {0, 2, 3, 1};
        out->Resize(std::vector<int64_t>{
            origin_dims[0], origin_dims[2], origin_dims[3], origin_dims[1]});
        break;
      default:
        CHECK(0) << "Unsupport dim in mlu layout nchw to nhwc";
    }

    LayoutTransCompute<lite::TargetType::kX86,
                       typename subgraph::mlu::MLUTypeTraits<Precision>::type>(
        x_ndims, context, *x, out, axis);

    if (x_ndims > 2) {
      out->Resize(origin_dims);
    }
  }

  std::string doc() const override {
    return "Mlu layout transform nchw to nhwc";
  }
};

template <PrecisionType Precision>
class LayoutNhwcToNchwCompute
    : public KernelLite<TARGET(kX86), Precision, DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::LayoutParam;

  void Run() override {
    auto& param = this->template Param<param_t>();
    auto* x = param.x;
    auto* out = param.y;
    out->template mutable_data<
        typename subgraph::mlu::MLUTypeTraits<Precision>::type>();
    auto& context = this->ctx_->template As<X86Context>();

    TensorLite tmp_t;
    tmp_t.ShareDataWith(*x);

    const auto x_dims = x->dims().Vectorize();
    auto x_ndims = param.x->dims().size();
    std::vector<int> axis;
    switch (x_ndims) {
      case 2:
        axis = {0, 1};
        break;
      case 3:
        tmp_t.Resize(std::vector<int64_t>{x_dims[0], x_dims[2], x_dims[1]});
        axis = {0, 2, 1};
        break;
      case 4:
        tmp_t.Resize(
            std::vector<int64_t>{x_dims[0], x_dims[2], x_dims[3], x_dims[1]});
        axis = {0, 3, 1, 2};
        break;
      default:
        CHECK(0) << "Unsupport dim in mlu layout nhwc to nchw";
    }

    LayoutTransCompute<lite::TargetType::kX86,
                       typename subgraph::mlu::MLUTypeTraits<Precision>::type>(
        x_ndims, context, tmp_t, out, axis);
  }

  std::string doc() const override {
    return "Mlu layout transform nhwc to nchw";
  }
};

template <PrecisionType Precision, DataLayoutType in_layout>
class LayoutComputeMlu
    : public KernelLite<TARGET(kMLU), Precision, DATALAYOUT(kNHWC)> {
 public:
  using param_t = operators::LayoutParam;

  void Run() override {
    auto& param = this->template Param<param_t>();
    auto* x = param.x;
    auto* y = param.y;
    auto in_dims = x->dims().Vectorize();
    y->template mutable_data<
        typename subgraph::mlu::MLUTypeTraits<Precision>::type>();
    auto& context = this->ctx_->template As<MLUContext>();

    // key to map op
    std::vector<int> ishape;
    std::transform(in_dims.cbegin(),
                   in_dims.cend(),
                   std::back_inserter(ishape),
                   [](DDim::value_type in) { return static_cast<int>(in); });

    // find compiled instruction at ishape
    auto op_iter = inst_map_.find(ishape);
    if (op_iter == inst_map_.end()) {
      auto res =
          inst_map_.insert({ishape, CompileOperator(&param, &context, ishape)});
      CHECK(res.second);
      op_iter = res.first;
    }

    // prepare param
    auto exec_queue = context.exec_queue();
    cnrtInvokeFuncParam_t forward_param = context.forward_param();
    int data_param = 1;
    forward_param.data_parallelism = &data_param;
    u32_t affinity = context.affinity();
    forward_param.affinity = &affinity;
    forward_param.end = CNRT_PARAM_END;

    // get input and output
    auto mem_size = x->memory_size();
    y->set_precision(Precision);
    const void* input = x->template data<
        typename subgraph::mlu::MLUTypeTraits<Precision>::type>();
    void* output = y->mutable_data(TARGET(kMLU), mem_size);

    // compute op
    CNML_CALL(cnmlComputeNdTransposeProOpForward(op_iter->second->cnml_op,
                                                 const_cast<void*>(input),
                                                 output,
                                                 &forward_param,
                                                 exec_queue));
  }

  std::string doc() const override { return "Mlu layout transform"; }

 private:
  std::shared_ptr<MLUOperator> CompileOperator(param_t* param,
                                               MLUContext* ctx,
                                               std::vector<int> dims) {
    VLOG(4) << "compile layout operator";
    // get transpose axis
    std::vector<int> axis;
    std::vector<int> in_dims, out_dims;
    if (in_layout == DATALAYOUT(kNCHW)) {
      VLOG(4) << "trans layout from NCHW to NHWC";
      axis = subgraph::mlu::GetAxisNCHW2NHWC<int>(dims.size());
      in_dims = dims;
      out_dims = subgraph::mlu::DimNCHW2NHWC(dims);
    } else {
      VLOG(4) << "trans layout from NHWC to NCHW";
      axis = subgraph::mlu::GetAxisNHWC2NCHW<int>(dims.size());
      in_dims = subgraph::mlu::DimNCHW2NHWC(dims);
      out_dims = dims;
    }

    // prepare op and io tensor
    auto op = std::make_shared<MLUOperator>();
    op->input_tensors.emplace_back();
    op->output_tensors.emplace_back();

    int* dim_strides = nullptr;
    CNML_CALL(cnmlCreateTensor_V2(&op->input_tensors[0], CNML_TENSOR));
    CNML_CALL(cnmlSetTensorShape_V2(
        op->input_tensors[0], in_dims.size(), in_dims.data(), dim_strides));
    CNML_CALL(cnmlSetTensorDataType(
        op->input_tensors[0],
        subgraph::mlu::MLUTypeTraits<Precision>::cnml_type));

    CNML_CALL(cnmlCreateTensor_V2(&op->output_tensors[0], CNML_TENSOR));
    CNML_CALL(cnmlSetTensorShape_V2(
        op->output_tensors[0], out_dims.size(), out_dims.data(), dim_strides));
    CNML_CALL(cnmlSetTensorDataType(
        op->output_tensors[0],
        subgraph::mlu::MLUTypeTraits<Precision>::cnml_type));

    cnmlNdTransposeOpParam_t transpose_param;
    CNML_CALL(cnmlCreateNdTransposeOpParam(
        &transpose_param, axis.data(), axis.size()));
    CNML_CALL(cnmlCreateNdTransposeProOp(&op->cnml_op,
                                         op->input_tensors[0],
                                         op->output_tensors[0],
                                         transpose_param));
    CNML_CALL(cnmlDestroyNdTransposeOpParam(&transpose_param));

    CNML_CALL(cnmlSetBaseOpCorenum(op->cnml_op, ctx->MLUCoreNumber()));
    CNML_CALL(cnmlSetBaseOpCoreVersion(op->cnml_op, ctx->MLUCoreVersion()));
    CNML_CALL(cnmlCompileBaseOp_V2(op->cnml_op));
    return op;
  }
  std::map<std::vector<int>, std::shared_ptr<MLUOperator>> inst_map_;
};

template <PrecisionType precision>
using LayoutNHWC2NCHW = LayoutComputeMlu<precision, DATALAYOUT(kNHWC)>;
template <PrecisionType precision>
using LayoutNCHW2NHWC = LayoutComputeMlu<precision, DATALAYOUT(kNCHW)>;

}  // namespace mlu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
