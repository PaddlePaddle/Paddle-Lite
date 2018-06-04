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
class AclConvOp : public acl::ACLOperator {
 public:
  AclConvOp(){
      this->force_bypass_acl_path_= bypass_acl_class_layer & 
                                    FLAGS_ENABLE_ACL_CONV;
  }
  ~AclConvOp() = default;
  AclConvOp(const AclConvOp&) = delete;
  AclConvOp& operator=(const AclConvOp&) = delete;
  AclConvOp(AclConvOp&&) = delete;
  AclConvOp& operator=(AclConvOp&&) = delete;

  acl::AclParameters& getargs() {
      return args;
  }
  void InitAclLayer(const ConvParam &param) {
    arm_compute::TensorShape input_shape(
      args.in_cols, args.in_rows, args.in_depth, args.batch);
    arm_compute::TensorShape output_shape(
      args.out_cols, args.out_rows, args.out_depth, args.out_num);
    arm_compute::TensorShape weights_shape(
      args.filter_cols, args.filter_rows, args.in_depth/args.num_group, args.out_depth);
    //arm_compute::TensorShape biases_shape(args.out_depth);
    arm_compute::PadStrideInfo conv_info(args.stride_cols, args.stride_rows,
      args.pad_cols, args.pad_rows, arm_compute::DimensionRoundingType::FLOOR);

    if (is_operator_init_done(input_shape)) return;
    set_operator_init_done();
    this->force_bypass_acl_path_=false;

    check_direct_conv();
    //[kernel_x, kernel_y, IFM, OFM]
    new_tensor(weights(),weights_shape,args.weight_data);
    //[OFM]
    //if (args.biases_data) {
    //    new_tensor(biases(),biases_shape,args.biases_data);
    //}

    group() = args.num_group;

    //[width, height, IFM]
    new_tensor(input(),input_shape,args.input_data);
    //[width, height, OFM]
    new_tensor(output(),output_shape,args.output_data);

    acl_configure(conv,this,conv_info);
  }

  void RunAcl(void* input, void* output) {
    acl::ACLOperator::acl_run(input, output);
  }
  bool Bypass_acl(const ConvParam &param) {
    bool bypass_acl = false;
    AclParametersByContext(param);
    //for performance, more groups impact GPU performance
    if (this->force_bypass_acl_path_ || args.num_group>=5) {
        bypass_acl = true;
    }
    if (args.dim >2) {
        bypass_acl = true;
    }
    if(args.dilated) {
        bypass_acl = true;
    }
    return bypass_acl;
  }
private:
  void check_direct_conv(){
      bool use_direct_conv=false;
      const char* pDirectConv;
      pDirectConv = getenv ("DIRECTCONV");
      if (pDirectConv){
        unsigned int bdirectconv;
        sscanf(pDirectConv,"%i", &bdirectconv);
        if(bdirectconv != use_direct_conv){
            use_direct_conv = bdirectconv;
            printf("DIRECTCONV<%s>\n", pDirectConv);
            printf("DIRECTCONV: %x\n", use_direct_conv);
        }
      }
      int pad_data[2], kernel[2]; 
      pad_data[1] = args.pad_rows;
      pad_data[0] = args.pad_cols;
      kernel[1] = args.filter_rows;
      kernel[0] = args.filter_cols;
      if (use_direct_conv && 
          ((kernel[0] == 1 && kernel[1] == 1 && pad_data[0] == 0 && pad_data[1] == 0) ||
            (kernel[0] == 3 && kernel[1] == 3 && pad_data[0] <= 1 && pad_data[1] <= 1 ) )) {
          setConvMethod(); //NEDirectConvolutionLayer only for 1x1 and 3x3
      }
      #ifdef USE_OPENGLES
          if (param_.num_group>1) {
             bypass_acl_class_layer|=FLAGS_ENABLE_ACL_CONV;
          }
      #endif

  }

  void AclParametersByContext(const ConvParam &param){
    const Tensor *input = param.Input();
    Tensor filter = *param.Filter();
    Tensor *output = param.Output();

    int groups = param.Groups();
    std::vector<int> strides = param.Strides();
    std::vector<int> paddings = param.Paddings();
    std::vector<int> dilations = param.Dilations();

    const T* input_data = input->data<T>();
    T* output_data = output->mutable_data<T>();
    const T* weight_data = filter.data<T>();

    args.input_data = (void*)input_data;
    args.output_data = (void*)output_data;
    args.weight_data = (void*)weight_data;
    args.biases_data = nullptr;

    // try {
    //     bias = context.Input<framework::Tensor>("Bias");
    // } catch (const std::exception& e) {
    // }
    // if (bias) {
    //     const T* biases_data = bias->data<T>();
    //     args.biases_data = (void*)biases_data;
    // }

    args.num_group = groups;

    args.dilation_rows = dilations[0];
    args.dilation_cols = dilations[1];
    if (dilations[0]!= 1 || dilations[1]!= 1){
      args.dilated = true;
    }

    // NCHW
    args.batch    = input->dims()[0];
    args.in_depth = input->dims()[1];
    args.in_rows  = input->dims()[2];
    args.in_cols  = input->dims()[3];
    std::cout <<"In N: " << args.batch << " C: " <<  args.in_depth
      << " H: " << args.in_rows << " W: " << args.in_cols << "\n";
    // NCHW
    args.out_num = output->dims()[0];
    args.out_depth = output->dims()[1];
    args.out_rows  = output->dims()[2];
    args.out_cols  = output->dims()[3];
    std::cout <<"Out N: " << static_cast<int>(output->dims()[0])
      << " C: " <<  args.out_depth
      << " H: " << args.out_rows << " W: " << args.out_cols << "\n";
    // MCHW = OIHW
    args.filter_rows  = filter.dims()[2];
    args.filter_cols  = filter.dims()[3];
    std::cout <<"Filter O: " << static_cast<int>(filter.dims()[0])
      << " I: " <<  static_cast<int>(filter.dims()[1])
      << " H: " << args.filter_rows << " W: " << args.filter_cols << "\n";

    // strides(h_stride, w_stride)
    args.stride_rows = strides[0];
    args.stride_cols = strides[1];
    std::cout <<"Stride H: " << args.stride_rows << " W: " << args.stride_cols << "\n";

    // paddings(h_pad, w_pad)
    args.pad_rows = paddings[0];
    args.pad_cols = paddings[1];
    std::cout <<"Pad H: " << args.pad_rows << " W: " << args.pad_cols << "\n";
  }
  acl::AclParameters args;

};

template <typename DeviceType, typename T>
class AclConvKernel : public framework::OpKernelBase<DeviceType, ConvParam> {
 public:
  bool Bypass_acl(const ConvParam &param) const {
      AclConvOp<DeviceType, T>* acl_op = reinterpret_cast<AclConvOp<DeviceType, T>*>(this->GetAclOp());
      if (acl_op == nullptr) {
        acl_op = new AclConvOp<DeviceType, T>();
        this->SetAclOp((void*)acl_op, (void*)this);
      }
      return acl_op->Bypass_acl(param);
  }

  void Compute(const ConvParam &param) const override {
    AclConvOp<DeviceType, T>* acl_op = reinterpret_cast<AclConvOp<DeviceType, T>*>(this->GetAclOp());
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
    acl_op->RunAcl((void*)input_data, (void*)output_data);

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
