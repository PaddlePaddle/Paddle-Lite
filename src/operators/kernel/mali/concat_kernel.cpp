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

#ifdef CONCAT_OP

#include "operators/kernel/concat_kernel.h"
#ifdef PADDLE_MOBILE_MALI_GPU
#include "acl_operator.h"
#include "framework/operator.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename DeviceType, typename T>
class AclConcatOp : public acl::ACLOperator {
 public:
  AclConcatOp() {
    this->force_bypass_acl_path_ =
        bypass_acl_class_layer & FLAGS_ENABLE_ACL_CONCAT;
  }
  ~AclConcatOp() = default;
  AclConcatOp(const AclConcatOp&) = delete;
  AclConcatOp& operator=(const AclConcatOp&) = delete;
  AclConcatOp(AclConcatOp&&) = delete;
  AclConcatOp& operator=(AclConcatOp&&) = delete;

  acl::AclParameters& getargs() { return args; }

  void InitAclLayer(const ConcatParam<DeviceType>& param) {
    setTargetHint(acl::TargetHint::OPENCL);
    const std::vector<framework::LoDTensor*>* input_data = &args.in_tensor;
    arm_compute::TensorShape output_shape(args.out_cols, args.out_rows,
                                          args.out_depth, args.batch);

    if (is_operator_init_done(output_shape)) return;
    set_operator_init_done();
    this->force_bypass_acl_path_ = false;
    T type;

    for (int i = 0; i < input_data->size(); i++) {
      int in_batch = (*input_data)[i]->dims()[0];
      int in_channels = (*input_data)[i]->dims()[1];
      int in_width = (*input_data)[i]->dims()[2];
      int in_height = (*input_data)[i]->dims()[3];
      arm_compute::TensorShape in_shape(in_width, in_height, in_channels);

      new_tensor(cinput(i), in_shape,
                 acl::InputdataPtr(this, args.in_tensor, type, i));
    }

    //[width, height, OFM]
    new_tensor(output(), output_shape, args.output_data);

    acl_configure(concat, this, input_data->size());
  }

  void RunAcl(const std::vector<framework::LoDTensor*>& input, void* output) {
    T type;
    acl::acl_run(this, input, output, type);
  }
  bool Bypass_acl(const ConcatParam<DeviceType>& param) {
    bool bypass_acl = false;
    AclParametersByContext(param);
    InitAclLayer(param);
    // for performance, more groups impact GPU performance
    if (this->force_bypass_acl_path_ || !args.is_channel_concat) {
      bypass_acl = true;
    }
    return bypass_acl;
  }

 private:
  void AclParametersByContext(const ConcatParam<DeviceType>& param) {
    auto inputs = param.Inputs();
    auto* output = param.Out();
    int64_t axis = param.Axis();

    T* output_data = output->mutable_data<T>();

    args.is_channel_concat = (axis == 1);
    args.in_tensor = inputs;
    args.output_data = (void*)output_data;

    args.batch = output->dims()[0];
    args.out_depth = output->dims()[1];
    args.out_rows = output->dims()[2];
    args.out_cols = output->dims()[3];
  }
  acl::AclParameters args;
};

template <>
bool ConcatKernel<GPU_MALI, float>::Init(ConcatParam<GPU_MALI>* param) {
  AclConcatOp<GPU_MALI, float>* acl_op =
      reinterpret_cast<AclConcatOp<GPU_MALI, float>*>(this->GetAclOp());
  if (acl_op == nullptr) {
    acl_op = new AclConcatOp<GPU_MALI, float>();
    this->SetAclOp((void*)acl_op, (void*)this);
  }
  if (acl_op->Bypass_acl(*param)) {
    std::cout << "init acl failed" << std::endl;
    return false;
  }
  return true;
}

template <>
void ConcatKernel<GPU_MALI, float>::Compute(
    const ConcatParam<GPU_MALI>& param) {
  std::cout << "init acl" << std::endl;
  AclConcatOp<GPU_MALI, float>* acl_op =
      reinterpret_cast<AclConcatOp<GPU_MALI, float>*>(this->GetAclOp());
  if (acl_op == nullptr) {
    return;
  }
  acl::AclParameters& args = acl_op->getargs();
  acl_op->RunAcl(args.in_tensor, args.output_data);
}

template class ConcatKernel<GPU_MALI, float>;
}  // namespace operators
}  // namespace paddle_mobile

#endif
#endif
