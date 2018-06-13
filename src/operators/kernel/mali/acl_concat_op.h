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
class AclConcatOp : public acl::ACLOperator {
 public:
  AclConcatOp(){
      this->force_bypass_acl_path_= bypass_acl_class_layer &
                                    FLAGS_ENABLE_ACL_CONCAT;
  }
  ~AclConcatOp() = default;
  AclConcatOp(const AclConcatOp&) = delete;
  AclConcatOp& operator=(const AclConcatOp&) = delete;
  AclConcatOp(AclConcatOp&&) = delete;
  AclConcatOp& operator=(AclConcatOp&&) = delete;

  acl::AclParameters& getargs() {
      return args;
  }

  void InitAclLayer(const ConcatParam &param) {
    setTargetHint(acl::TargetHint::OPENCL);
    const std::vector<framework::LoDTensor*>* input_data = &args.in_tensor;
    arm_compute::TensorShape output_shape(
      args.out_cols, args.out_rows, args.out_depth, args.batch);

    if (is_operator_init_done(output_shape)) return;
    set_operator_init_done();
    this->force_bypass_acl_path_=false;
    T type;

    for (int i = 0; i < input_data->size(); i++) {
      const T* idata= (*input_data)[i]->data<T>();
      const T* pdata= (*input_data)[i]->data<T>();
      int in_batch = (*input_data)[i]->dims()[0];
      int in_channels = (*input_data)[i]->dims()[1];
      int in_width = (*input_data)[i]->dims()[2];
      int in_height = (*input_data)[i]->dims()[3];
      arm_compute::TensorShape in_shape(in_width, in_height,in_channels);

      new_tensor(cinput(i),in_shape,acl::InputdataPtr(this,args.in_tensor,type,i));
    }

    //[width, height, OFM]
    new_tensor(output(),output_shape,args.output_data);

    acl_configure(concat,this,input_data->size());
  }

  void RunAcl(const std::vector<framework::LoDTensor*>& input, void* output) {
    T type;
    acl::acl_run(this, input, output, type);
  }
  bool Bypass_acl(const ConcatParam &param) {
    bool bypass_acl = false;
    AclParametersByContext(param);
    //for performance, more groups impact GPU performance
    if (this->force_bypass_acl_path_ || !args.is_channel_concat) {
        bypass_acl = true;
    }
    return bypass_acl;
  }
private:
  void AclParametersByContext(const ConcatParam &param){
    auto inputs = param.Inputs();
    auto *output = param.Out();
    int64_t axis = param.Axis();

    T* output_data = output->mutable_data<T>();

    args.is_channel_concat = (axis == 1);
    args.in_tensor = inputs;
    args.output_data = (void*)output_data;

    args.batch = output->dims()[0];
    args.out_depth = output->dims()[1];
    args.out_rows  = output->dims()[2];
    args.out_cols  = output->dims()[3];
  }
  acl::AclParameters args;

};

template <typename DeviceType, typename T>
class AclConcatKernel : public framework::OpKernelBase<DeviceType, ConcatParam> {
 public:
  bool Bypass_acl(const ConcatParam &param) const {
      AclConcatOp<DeviceType, T>* acl_op = reinterpret_cast<AclConcatOp<DeviceType, T>*>(this->GetAclOp());
      if (acl_op == nullptr) {
        acl_op = new AclConcatOp<DeviceType, T>();
        this->SetAclOp((void*)acl_op, (void*)this);
      }
      return acl_op->Bypass_acl(param);
  }

  void Compute(const ConcatParam &param) const override {
    AclConcatOp<DeviceType, T>* acl_op = reinterpret_cast<AclConcatOp<DeviceType, T>*>(this->GetAclOp());
    if (acl_op == nullptr) {
        return;
    }
    acl::AclParameters& args = acl_op->getargs();
    //const T* input_data = (const T*)args.input_data;

    std::vector<framework::LoDTensor*> temp_data = args.in_tensor;
    const T* output_data = (const T*)args.output_data;
    //const T* input_data = (const T*)temp_p;

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
    acl_op->RunAcl(temp_data, (void*)output_data);

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
}  // namespace paddle
#endif //USE_ACL
