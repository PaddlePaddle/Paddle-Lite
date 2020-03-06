
#pragma once

// #include <arm_neon.h>
#include <vector>

#include "../pe.hpp"
#include "../pe_params.hpp"
#include "concat_pe.hpp"
#include "conv_pe.hpp"
#include "transposed_conv_process.hpp"
#include "scale_pe.hpp"

namespace paddle_mobile {
namespace zynqmp {

class TransposedConvPE : public PE {
 public:
  bool init() {
    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);

    return true;
  }

  void apply() {
    // fill_transposed_split_arg(param_);
    Shape& input_shape = param_.input->shape();
    int padded_height = input_shape.height() + (input_shape.height() - 1) * (param_.strides[0] - 1);
    int padded_width = input_shape.width() + (input_shape.width() - 1) * (param_.strides[1] - 1);

    Shape padded_shape(NCHW, {
      input_shape.num(), 
      input_shape.channel(),
      padded_height,
      padded_width
    });

    // std::cout << "iw:" << input_shape.width()<< "ih:" << input_shape.height()
    //      << " ic:" << input_shape.channel() << std::endl;
    // std::cout << "ph:" << padded_height << " pw:" << padded_width << std::endl;

    int p = param_.filter->shape().height() - param_.paddings[0] - 1;
    int ph =  param_.filter->shape().height() - param_.paddings[0] - 1;
    int pw =  param_.filter->shape().width() - param_.paddings[1] - 1;
    // std::cout << " p:" << p << std::endl;
    
    padded_input_.mutableData<float16>(FP16, padded_shape);

    ConvParam& conv_param = pe_.param();
    conv_param.input = &padded_input_;
    conv_param.output = param_.output;
    conv_param.filter = &filter_;

    conv_param.strides = {1, 1};
    conv_param.paddings = {ph, pw};

    conv_param.kernelSize = param_.kernelSize;
    conv_param.dilations = {1, 1};

    conv_param.activeParam.type = param_.activeParam.type;

    // std::cout << "padding:" << p << std::endl;
    // exit(-1);

    // std::cout << "previous filter's shape layout - num:" << param_.filter->shape().num() << " channel:" << param_.filter->shape().channel() << std::endl;

    convert_cnhw_to_nchw(param_.filter, &filter_);
    // filter_.saveToFile("before_inverse_filter", true);
    inverse_filter(&filter_);

    // std::cout << "filter's shape layout - num:" << filter_.shape().num() << " channel:" << filter_.shape().channel() << std::endl;

    // filter_.saveToFile("after_inverse_filter", true);

    // fill_scale_bias_const(&conv_param);


    conv_param.scale()->mutableData<float>(FP32, param_.scale()->shape());
    conv_param.scale()->copyFrom(param_.scale());

    conv_param.bias()->mutableData<float>(FP32, param_.bias()->shape());
    conv_param.bias()->copyFrom(param_.bias());

    // filter_.saveToFile("filter", true);

    pe_.init();
    pe_.apply();
  }

  template<typename T>
  void pad_input() {
    param_.input->syncToCPU();
    T* input_data = param_.input->data<T>();

    int channel = param_.input->shape().channel();
    int in_wc = param_.input->shape().width() * channel;

    int o_wc = padded_input_.shape().width() * channel;

    // int s_h = param_.strides[1] - 1;
    // int s_w = param_.strides[0] - 1;

    T* data = padded_input_.data<T>();
    int oh = param_.input->shape().height();
    int ow = param_.input->shape().width();

    memset(data, 0, padded_input_.memorySize());

    for (int h = 0; h < oh; h++) {
      for (int w = 0; w < ow; w++) {
        T* src = input_data + h * in_wc + w * channel;
        T* dst = data + (h) * param_.strides[0] * o_wc + (w) * (param_.strides[1]) * channel;
        memcpy(dst, src, channel * sizeof(T));
      }
    }

    padded_input_.flush();
    // param_.input->saveToFile("in", true);
    // padded_input_.saveToFile("padded_input", true);
    padded_input_.copyScaleFrom(param_.input);
  };

  bool dispatch() {
    // std::cout << "dispatch \n";
    pad_input<float16>();
    // std::cout << "after pad \n";
    return pe_.dispatch();
  }

  ConvParam& param() { return param_; }

 private:
  ConvParam param_;
  ConvPE pe_;
  Tensor padded_input_;
  Tensor filter_;
  InplaceArgs inplace_ = {0};
  ActiveParamterArgs activeParamterArgs;
};

}  // namespace zynqmp
}  // namespace paddle_mobile
