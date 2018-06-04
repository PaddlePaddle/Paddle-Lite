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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "acl_operator.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class AclActCtx : public acl::ACLOperator {
 public:
  AclActCtx(){
      this->force_bypass_acl_path_ = bypass_acl_class_layer &
                                     FLAGS_ENABLE_ACL_RELU & 
                                     FLAGS_ENABLE_ACL_TANH;
  }
  ~AclActCtx() = default;
  AclActCtx(const AclActCtx&) = delete;
  AclActCtx& operator=(const AclActCtx&) = delete;
  AclActCtx(AclActCtx&&) = delete;
  AclActCtx& operator=(AclActCtx&&) = delete;

  acl::AclParameters& getargs() {
      return args;
  }
  void InitAclLayer(const framework::ExecutionContext& context) {
    //const unsigned int count = args.in_cols* args.in_rows* args.in_depth* args.batch;
    arm_compute::TensorShape input_shape(
      args.in_cols*args.in_rows);
    arm_compute::TensorShape output_shape(
      args.in_cols*args.in_rows);
    //arm_compute::TensorShape weights_shape(
      //args.filter_cols, args.filter_rows, args.in_depth, args.out_depth);
    //arm_compute::TensorShape biases_shape(args.out_depth);
    arm_compute::ActivationLayerInfo::ActivationFunction type;
    int type_cls;
    if (args.act_type == "relu") type_cls = 1;
    if (args.act_type == "tanh") type_cls = 2;
    if (args.act_type == "softplus") type_cls = 3;
    switch(type_cls){
            default:
            case 1:
                type=arm_compute::ActivationLayerInfo::ActivationFunction::RELU;
                break;
            case 2:
                type=arm_compute::ActivationLayerInfo::ActivationFunction::TANH;
                break;
            case 3:
                type=arm_compute::ActivationLayerInfo::ActivationFunction::SOFT_RELU;
                break;
    }
    arm_compute::ActivationLayerInfo act_info(type);
    if(type== arm_compute::ActivationLayerInfo::ActivationFunction::TANH)
          act_info=arm_compute::ActivationLayerInfo(type,1.0,1.0);


    if (is_operator_init_done(input_shape)) return;
    set_operator_init_done();
    this->force_bypass_acl_path_=false;

    //[width, height, IFM]
    new_tensor(input(),input_shape,args.input_data);
    //[width, height, OFM]
    new_tensor(output(),output_shape,args.output_data);

    acl_configure(activation,this,act_info);
  }

  void RunAcl(void* input, void* output) {
    acl::ACLOperator::acl_run(input, output);
  }
  bool Bypass_acl(const framework::ExecutionContext& context) {
    bool bypass_acl = false;
    AclParametersByContext(context);
    //for performance, more groups impact GPU performance
    if (this->force_bypass_acl_path_) {
      bypass_acl = true;
    }
    if (args.act_type != "relu" && args.act_type != "tanh" &&
      args.act_type != "softplus") {
      bypass_acl = true;
    }
    else if ((args.act_type == "relu" && !FLAGS_ENABLE_ACL_RELU) ||
      (args.act_type == "tanh" && !FLAGS_ENABLE_ACL_TANH)) {
      bypass_acl = true;
    }

    return bypass_acl;
  }
private:
  void AclParametersByContext(const framework::ExecutionContext& context){
    auto act_type = context.op().Type();
    auto& X = detail::Ref(context.Input<framework::Tensor>("X"),
                          "Cannot get input tensor X, variable name = %s",
                          context.op().Input("X"));

    auto& Out = detail::Ref(context.Output<framework::Tensor>("Out"),
                            "Cannot get output tensor Out, variable name = %s",
                            context.op().Output("Out"));

    //auto x = framework::EigenVector<T>::Flatten(X);
    //auto out = framework::EigenVector<T>::Flatten(Out);
    const T* input_data = X.data<T>();
    T* output_data = Out.mutable_data<T>(context.GetPlace());

    args.input_data = (void*)input_data;
    args.output_data = (void*)output_data;

    for (int i = 0 ; i < X.dims().size(); i++){
      std::cout << X.dims()[i] <<std::endl;
    }

    std::cout << act_type <<std::endl;
    args.act_type = act_type;
    args.in_cols = X.dims()[0];
    args.in_rows = X.dims()[1];

  }
  acl::AclParameters args;

};

template <typename DeviceContext, typename T>
class AclActivationKernel : public framework::OpKernel<T> {
 public:
  bool Bypass_acl(const framework::ExecutionContext& context) const {
      AclActCtx<DeviceContext, T>* acl_ctx = reinterpret_cast<AclActCtx<DeviceContext, T>*>(context.GetAclCtx());
      if (acl_ctx == nullptr) {
        acl_ctx = new AclActCtx<DeviceContext, T>();
        framework::ExecutionContext* ctx = 
          const_cast<framework::ExecutionContext*>(&context);
        ctx->SetAclCtx((void*)acl_ctx);
      }
      return acl_ctx->Bypass_acl(context);
  }

  void Compute(const framework::ExecutionContext& context) const override {

    AclActCtx<DeviceContext, T>* acl_ctx = reinterpret_cast<AclActCtx<DeviceContext, T>*>(context.GetAclCtx());
    if (acl_ctx == nullptr) {
        return;
    }

    acl::AclParameters& args = acl_ctx->getargs();
    const T* input_data = (const T*)args.input_data;
    const T* output_data = (const T*)args.output_data;
#if 0
    std::cout << "Input: " << std::endl;
    for (int i = 0; i < X.dims()[0]; i++) {
      for (int m = 0; m < X.dims()[1]; m++) {
        std::cout << " " << *input_data++;
        std::cout << std::endl;
      }
        std::cout << std::endl;
    }

    std::cout << "Output: " << std::endl;
    for (int i = 0; i < X.dims()[0]; i++) {
      for (int m = 0; m < X.dims()[1]; m++) {
        std::cout << " " << *output_data++;
        std::cout << std::endl;
      }
        std::cout << std::endl;
    }
#endif
    acl_ctx->InitAclLayer(context);
    acl_ctx->RunAcl((void*)input_data, (void*)output_data);


    

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

#if 0
  bool AclCheckParamsInternal(const AclConv2DArgs& args) {
#if 0
    LOG(INFO) << this << " Conv2D: in_depth = " << args.in_depth
            << ", input_cols = " << args.in_cols
            << ", filter_cols = " << args.filter_cols
            << ", input_rows = " << args.in_rows
            << ", filter_rows = " << args.filter_rows
            << ", stride_rows = " << args.stride_rows
            << ", stride_cols = " << args.stride_cols
            << ", out_depth = " << args.out_depth
            << ", pad_rows = " << args.pad_rows
            << ", pad_cols = " << args.pad_cols
            << ", batch = " << args.batch
            << ", round_type: " << (int)round_type_;
#endif
    if (round_type_ == arm_compute::DimensionRoundingType::CEIL
        || round_type_ == arm_compute::DimensionRoundingType::FLOOR)
    return true;

    int conv_w = -1, conv_h= -1;
    std::tie(conv_w, conv_h) = arm_compute::scaled_dimensions(
      args.in_cols, args.in_rows,
      args.filter_cols, args.filter_rows,
      arm_compute::PadStrideInfo(
        args.stride_cols, args.stride_rows,
        args.pad_cols, args.pad_rows,
        arm_compute::DimensionRoundingType::CEIL));
    if (conv_w == args.out_cols && conv_h == args.out_rows) {
      round_type_ = arm_compute::DimensionRoundingType::CEIL;
      return true;
    }

    std::tie(conv_w, conv_h) = arm_compute::scaled_dimensions(
      args.in_cols, args.in_rows,
      args.filter_cols, args.filter_rows,
      arm_compute::PadStrideInfo(
        args.stride_cols, args.stride_rows,
        args.pad_cols, args.pad_rows,
      arm_compute::DimensionRoundingType::FLOOR));

    if (conv_w == args.out_cols && conv_h == args.out_rows) {
      round_type_ = arm_compute::DimensionRoundingType::FLOOR;
      return true;
    }

    LOG(ERROR) << "Acl(" << conv_w << "," << conv_h << ")"
      << " != TF(" << args.out_cols << "," << args.out_rows << ")";

    return false;
  }

    // filter_shape_vec: {k_o, k_i, k_h, k_w} or {k_o, k_i, k_d, k_h, k_w}
    std::vector<int64_t> filter_shape_vec(framework::vectorize(filter.dims()));
    // output_shape_vec: {o_n, o_c, o_h, o_w} or {o_n, o_c, o_d, o_h, o_w}
    std::vector<int64_t> output_shape_vec(framework::vectorize(output->dims()));

    auto& dev_ctx = context.template device_context<DeviceContext>();
    if (is_expand) {
      col.mutable_data<T>(col_shape, context.GetPlace());
      col_matrix.ShareDataWith(col);
      col_matrix.Resize(col_matrix_shape);
    }

    framework::DDim input_shape = framework::slice_ddim(
        input->dims(), 1, static_cast<int>(input->dims().size()));

    framework::DDim filter_matrix_shape = {filter.dims()[0],
                                           filter.numel() / filter.dims()[0]};
    filter.Resize(filter_matrix_shape);

    framework::DDim output_matrix_shape = {args
        output->dims()[1],
        output->numel() / (output->dims()[0] * output->dims()[1])};

    // convolution operator: im2col(or vol2col) + gemm
    int in_step = static_cast<int>(input->dims()[1]) / groups;
    int out_step = static_cast<int>(output->dims()[1]) / groups;

    math::Vol2ColFunctor<DeviceContext, T> vol2col;
    math::Im2ColFunctor<math::ColFormat::kCFO, DeviceContext, T> im2col;

    auto& dev_ctx = context.template device_context<DeviceContext>();
    for (int i = 0; i < batch_size; i++) {
      framework::Tensor in_batch = input->Slice(i, i + 1).Resize(input_shape);
      framework::Tensor out_batch = output->Slice(i, i + 1).Resize(output_matrix_shape);

      for (int g = 0; g < groups; g++) {
        framework::Tensor in_slice = in_batch.Slice(g * in_step, (g + 1) * in_step);

        if (!is_expand) {
          col.ShareDataWith(in_slice);
          col_matrix.ShareDataWith(col);
          col_matrix.Resize(col_matrix_shape);
        } else if (data_dim == 2U) {
          // im2col
          im2col(dev_ctx, in_slice, dilations, strides,
                 std::vector<int>{paddings[0], paddings[1], paddings[0],
                                  paddings[1]},
                 &col);
        } else if (data_dim == 3U) {
          // vol2col
          vol2col(dev_ctx, in_slice, dilations, strides, paddings, &col);
        }

        // gemm
        framework::Tensor out_slice = out_batch.Slice(g * out_step, (g + 1) * out_step);
        framework::Tensor filter_slice = filter.Slice(g * out_step, (g + 1) * out_step);
        math::matmul<DeviceContext, T>(dev_ctx, filter_slice, false, col_matrix,
                                       false, T(1.0), &out_slice, T(0.0));
      }
    }
#endif
};

}  // namespace operators
}  // namespace paddle
#endif //USE_ACL
