/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef POOL_OP

#pragma once

#include "operators/kernel/pool_kernel.h"
#ifdef PADDLE_MOBILE_MALI_GPU
#include "acl_operator.h"
#include "framework/operator.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename DeviceType, typename T>
class AclPoolOp : public acl::ACLOperator {
 public:
  AclPoolOp() {
    this->force_bypass_acl_path_ =
        bypass_acl_class_layer & FLAGS_ENABLE_ACL_POOLING;
  }
  ~AclPoolOp() = default;
  AclPoolOp(const AclPoolOp&) = delete;
  AclPoolOp& operator=(const AclPoolOp&) = delete;
  AclPoolOp(AclPoolOp&&) = delete;
  AclPoolOp& operator=(AclPoolOp&&) = delete;

  acl::AclParameters& getargs() { return args; }
  void InitAclLayer(const PoolParam<DeviceType>& param) {
    setTargetHint(acl::TargetHint::OPENCL);
    arm_compute::TensorShape input_shape(args.in_cols, args.in_rows,
                                         args.in_depth);
    arm_compute::TensorShape output_shape(args.out_cols, args.out_rows,
                                          args.out_depth);
    // arm_compute::TensorShape weights_shape(
    // args.filter_cols, args.filter_rows, args.in_depth, args.out_depth);
    // arm_compute::TensorShape biases_shape(args.out_depth);
    arm_compute::PoolingLayerInfo pool_info;

    if (args.pool_type == "max") {
      pool_info = arm_compute::PoolingLayerInfo(
          arm_compute::PoolingType::MAX, args.filter_rows,
          arm_compute::PadStrideInfo(args.stride_cols, args.stride_rows,
                                     args.pad_cols, args.pad_rows,
                                     arm_compute::DimensionRoundingType::CEIL));
    } else {
      pool_info = arm_compute::PoolingLayerInfo(
          arm_compute::PoolingType::AVG, args.filter_rows,
          arm_compute::PadStrideInfo(args.stride_cols, args.stride_rows,
                                     args.pad_cols, args.pad_rows,
                                     arm_compute::DimensionRoundingType::CEIL));
    }

    if (is_operator_init_done(input_shape)) return;
    set_operator_init_done();
    this->force_bypass_acl_path_ = false;

    //[width, height, IFM]
    new_tensor(input(), input_shape, args.input_data);
    //[width, height, OFM]
    new_tensor(output(), output_shape, args.output_data);

    acl_configure(pooling, this, pool_info);
  }

  void RunAcl(void* input, void* output) {
    acl::ACLOperator::acl_run(input, output);
  }
  bool Bypass_acl(const PoolParam<DeviceType>& param) {
    bool bypass_acl = false;
    AclParametersByContext(param);
    InitAclLayer(param);
    // for performance, more groups impact GPU performance
    if (this->force_bypass_acl_path_) {
      bypass_acl = true;
    }
    if (args.pool_type != "max" && args.pool_type != "avg") {
      bypass_acl = true;
    }
    if (args.filter_rows != args.filter_cols) {
      bypass_acl = true;
    }
    // if (args.filter_rows!=2 && args.filter_rows!=3) {
    //     bypass_acl = true;
    // }
    return bypass_acl;
  }

 private:
  void AclParametersByContext(const PoolParam<DeviceType>& param) {
    const Tensor* in_x = param.Input();
    Tensor* out = param.Output();
    std::string pooling_type = param.PoolingType();

    std::vector<int> ksize = param.Ksize();

    std::vector<int> strides = param.Strides();

    std::vector<int> paddings = param.Paddings();

    bool is_global_pooling = param.isGlobalPooling();

    const T* input_data = in_x->data<T>();
    T* output_data = out->mutable_data<T>();

    args.input_data = (void*)input_data;
    args.output_data = (void*)output_data;

    args.is_global_pool = is_global_pooling;
    args.pool_type = pooling_type;

    args.filter_rows = ksize[0];
    args.filter_cols = ksize[1];
    args.dim = ksize.size();

    // NCHW
    args.batch = in_x->dims()[0];
    args.in_depth = in_x->dims()[1];
    args.in_rows = in_x->dims()[2];
    args.in_cols = in_x->dims()[3];
    // std::cout <<"In N: " << args.batch << " C: " <<  args.in_depth
    //  << " H: " << args.in_rows << " W: " << args.in_cols << "\n";
    // NCHW
    // std::cout <<"Out N: " << static_cast<int>(output->dims()[0])
    //  << " C: " <<  args.out_depth
    //  << " H: " << args.out_rows << " W: " << args.out_cols << "\n";
    // MCHW = OIHW
    // std::cout <<"Filter O: " << static_cast<int>(filter->dims()[0])
    //  << " I: " <<  static_cast<int>(filter->dims()[1])
    //  << " H: " << args.filter_rows << " W: " << args.filter_cols << "\n";

    // strides(h_stride, w_stride)
    args.stride_rows = strides[0];
    args.stride_cols = strides[1];
    // std::cout <<"PoolingType: " << args.pool_type << "\n";
    // std::cout <<"Stride H: " << args.stride_rows << " W: " <<
    // args.stride_cols << "\n";

    // paddings(h_pad, w_pad)
    args.pad_rows = paddings[0];
    args.pad_cols = paddings[1];
    // std::cout <<"Pad H: " << args.pad_rows << " W: " << args.pad_cols <<
    // "\n";

    args.out_depth = args.in_depth;
    // args.out_rows = out->dims()[2];
    // args.out_cols = out->dims()[3];
    args.out_rows = static_cast<int>(ceil(static_cast<float>(args.in_rows +
                                                             2 * args.pad_rows -
                                                             args.filter_rows) /
                                          args.stride_rows)) +
                    1;
    args.out_cols = static_cast<int>(ceil(static_cast<float>(args.in_cols +
                                                             2 * args.pad_cols -
                                                             args.filter_cols) /
                                          args.stride_cols)) +
                    1;

    if (is_global_pooling) {
      args.filter_rows = args.in_rows;
      args.filter_cols = args.in_cols;
      args.pad_rows = 0;
      args.pad_cols = 0;
    }
  }
  acl::AclParameters args;
};

template <>
bool PoolKernel<GPU_MALI, float>::Init(PoolParam<GPU_MALI>* param) {
  AclPoolOp<GPU_MALI, float>* acl_op =
      reinterpret_cast<AclPoolOp<GPU_MALI, float>*>(this->GetAclOp());
  if (acl_op == nullptr) {
    acl_op = new AclPoolOp<GPU_MALI, float>();
    this->SetAclOp((void*)acl_op, (void*)this);
  }
  if (acl_op->Bypass_acl(*param)) {
    std::cout << "init acl failed" << std::endl;
    return false;
  }
  return true;
}

template <>
void PoolKernel<GPU_MALI, float>::Compute(const PoolParam<GPU_MALI>& param) {
  std::cout << "init acl" << std::endl;
  AclPoolOp<GPU_MALI, float>* acl_op =
      reinterpret_cast<AclPoolOp<GPU_MALI, float>*>(this->GetAclOp());
  if (acl_op == nullptr) {
    return;
  }
  acl::AclParameters& args = acl_op->getargs();
  const float* input_data = (const float*)args.input_data;
  const float* output_data = (const float*)args.output_data;
  for (int n = 0; n < args.batch; ++n) {
    acl_op->RunAcl((void*)input_data, (void*)output_data);
    input_data += args.in_depth * args.in_cols * args.in_rows;
    output_data += args.in_depth * args.out_cols * args.out_rows;
  }
}

template class PoolKernel<GPU_MALI, float>;
}  // namespace operators
}  // namespace paddle_mobile

#endif
#endif
