/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

// See docs in ../ops/nn_ops.cc.
#if USE_ACL == 1
#include <framework/operator.h>
#include <operators/op_param.h>
#include "acl_operator.h"

namespace paddle_mobile {
namespace operators {

template <typename DeviceType, typename T>
class AclLrnOp : public acl::ACLOperator {
 public:
  AclLrnOp() {
    this->force_bypass_acl_path_ =
        bypass_acl_class_layer & FLAGS_ENABLE_ACL_LRN;
  }
  ~AclLrnOp() = default;
  AclLrnOp(const AclLrnOp&) = delete;
  AclLrnOp& operator=(const AclLrnOp&) = delete;
  AclLrnOp(AclLrnOp&&) = delete;
  AclLrnOp& operator=(AclLrnOp&&) = delete;

  acl::AclParameters& getargs() { return args; }
  void InitAclLayer(const LrnParam& param) {
    setTargetHint(acl::TargetHint::OPENCL);
    arm_compute::TensorShape shape(
      args.in_cols, args.in_rows, args.in_depth);

    if (is_operator_init_done(shape)) return;
    set_operator_init_done();
    this->force_bypass_acl_path_=false;

    arm_compute::NormalizationLayerInfo norm_info(
      arm_compute::NormType::CROSS_MAP, args.nsize, args.alpha, args.beta, args.knorm);

    //[width, height, IFM]
    new_tensor(input(),shape,args.input_data);
    //[width, height, OFM]
    new_tensor(output(),shape,args.output_data);

    acl_configure(lrn,this,norm_info);
  }

  void RunAcl(void* input, void* output) {
    acl::ACLOperator::acl_run(input, output);
  }
  bool Bypass_acl(const LrnParam& param) {
    bool bypass_acl = false;
    AclParametersByContext(param);
    // for performance, more groups impact GPU performance
    if (this->force_bypass_acl_path_) {
      bypass_acl = true;
    }

    return bypass_acl;
  }

 private:
  void AclParametersByContext(const LrnParam& param) {
    const Tensor* in_x = param.InputX();
    Tensor* out = param.Out();

    int n = param.N();
    T alpha = param.Alpha();
    T beta = param.Beta();
    T k = param.K();

    const T* input_data = in_x->data<T>();
    T* output_data = out->mutable_data<T>();

    args.input_data = (void*)input_data;
    args.output_data = (void*)output_data;

    args.nsize = n;
    args.alpha = alpha;
    args.beta = beta;
    args.knorm = k;

    // NCHW
    args.batch    = in_x->dims()[0];
    args.in_depth = in_x->dims()[1];
    args.in_rows  = in_x->dims()[2];
    args.in_cols  = in_x->dims()[3];
    // std::cout
    //  << "Out C: " <<  args.out_depth
    //  << " H: " << args.out_rows << " W: " << args.out_cols << "\n";
  }
  acl::AclParameters args;
};

template <typename DeviceType, typename T>
class AclLrnKernel
    : public framework::OpKernelBase<DeviceType, LrnParam> {
 public:
  bool Bypass_acl(const LrnParam& param) const {
    AclLrnOp<DeviceType, T>* acl_op =
        reinterpret_cast<AclLrnOp<DeviceType, T>*>(this->GetAclOp());
    if (acl_op == nullptr) {
      acl_op = new AclLrnOp<DeviceType, T>();
      this->SetAclOp((void*)acl_op, (void*)this);
    }
    return acl_op->Bypass_acl(param);
  }

  void Compute(const LrnParam& param) const override {
    AclLrnOp<DeviceType, T>* acl_op =
        reinterpret_cast<AclLrnOp<DeviceType, T>*>(this->GetAclOp());
    if (acl_op == nullptr) {
      return;
    }

    acl::AclParameters& args = acl_op->getargs();
    const T* input_data = (const T*)args.input_data;
    const T* output_data = (const T*)args.output_data;
#if 0
    std::cout << "Input: " << std::endl;
    for (int i = 0; i < in_x->dims()[0]; i++) {
      for (int m = 0; m < in_x->dims()[1]; m++) {
        for (int j = 0; j < in_x->dims()[2]; j++) {
          for (int k = 0; k < in_x->dims()[3]; k++)
            std::cout << " " << *input_data++;
          std::cout << std::endl;
        }
        std::cout << std::endl;
      }
    }

    std::cout << "Output: " << std::endl;
    for (int i = 0; i < out->dims()[0]; i++) {
      for (int m = 0; m < out->dims()[1]; m++) {
        for (int j = 0; j < out->dims()[2]; j++) {
          for (int k = 0; k < out->dims()[3]; k++)
            std::cout << " " << *output_data++;
          std::cout << std::endl;
        }
        std::cout << std::endl;
      }
    }

    const T* weight_data = (const T*)args.weight_data;
    const framework::Tensor* filter = context.Input<framework::Tensor>("Filter");
   std::cout << "Input: " << std::endl;
  for(int i = 0; i < args.batch; i++) {
    for(int m = 0; m < args.in_depth; m++) {
      for(int j = 0; j < args.in_rows; j++) {
        for(int k = 0; k < args.in_cols; k++)
          std::cout << " " << *input_data++;
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
  }

  std::cout << "Filter: " << std::endl;
  for(int i = 0; i < static_cast<int>(filter->dims()[0]); i++) {
    for(int m = 0; m < static_cast<int>(filter->dims()[1]); m++) {
      for(int j = 0; j < args.filter_rows; j++) {
        for(int k = 0; k < args.filter_cols; k++)
          std::cout << " " << *weight_data++;
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
  }
#endif
    acl_op->InitAclLayer(param);
    for (int n = 0; n < args.batch; ++n) {
      acl_op->RunAcl((void*)input_data, (void*)output_data);
      input_data += args.in_depth * args.in_cols * args.in_rows;
      output_data += args.in_depth * args.in_cols * args.in_rows;
    }

#if 0
    std::cout << "Input: " << std::endl;
    for(int i = 0; i < args.batch; i++) {
      for(int m = 0; m < args.in_depth; m++) {
        for(int j = 0; j < args.in_rows; j++) {
          for(int k = 0; k < args.in_cols; k++)
            std::cout << " " << *input_data++;
          std::cout << std::endl;
        }
        std::cout << std::endl;
      }
    }

    std::cout << "Output: " << std::endl;
    for(int i = 0; i < args.batch; i++) {
      for(int m = 0; m < args.out_depth; m++) {
        for(int j = 0; j < args.out_rows; j++) {
          for(int k = 0; k < args.out_cols; k++)
            std::cout << " " << *output_data++;
          std::cout << std::endl;
        }
        std::cout << std::endl;
      }
    }
#endif
  }
};

}  // namespace operators
}  // namespace paddle_mobile
#endif  // USE_ACL
