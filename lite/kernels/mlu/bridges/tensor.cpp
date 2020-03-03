/*
All modification made by Cambricon Corporation: © 2018 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2018, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef USE_MLU
#include <glog/logging.h>
#include <vector>
#include "caffe/common.hpp"
#include "caffe/mlu/tensor.hpp"

namespace caffe {

void MLUTensorDesc::remember(const vector<int>& shape,
    cnmlTensorType_t tensor_type,
    BaseDataType cpu_dtype, BaseDataType mlu_dtype,
    cnmlDataOrder_t shape_order,
    vector<int>* dim_strides) {
  if (dim_strides != nullptr) {
    CHECK_EQ(shape.size(), dim_strides->size()) <<
      "Size of dim strides should equal size of tensor shape";
    dim_strides_ = *dim_strides;
    has_dim_strides_ = true;
    if (mlu_dtype == DT_UINT8 && shape[1] == 3)
      is_first_conv_input_tensor_ = true;
  }
  tensor_type_ = tensor_type;
  mlu_dtype_ = mlu_dtype;
  cpu_dtype_ = cpu_dtype;

  if (shape.size() > 0) {
    data_num_ = 1;
    for (auto dim : shape)
      data_num_ *= dim;
  }
  int size = 4;
  if (shape.size() > 4 || shape_order == CNML_ARRAY) {
    size = shape.size();
  }
  shape_.resize(size);
  if (shape.size() <= 4) {
    switch (shape_order) {
      case CNML_NCHW:
        shape_[0] = shape.size() > 0 ? shape[0] : 1;
        shape_[3] = shape.size() > 1 ? shape[1] : 1;
        shape_[1] = shape.size() > 2 ? shape[2] : 1;
        shape_[2] = shape.size() > 3 ? shape[3] : 1;
      break;
      case CNML_NCWH:
        shape_[0] = shape.size() > 0 ? shape[0] : 1;
        shape_[3] = shape.size() > 1 ? shape[1] : 1;
        shape_[2] = shape.size() > 3 ? shape[3] : 1;
        shape_[1] = shape.size() > 2 ? shape[2] : 1;
      break;
      case CNML_NHWC:
        shape_[0] = shape.size() > 0 ? shape[0] : 1;
        shape_[3] = shape.size() > 3 ? shape[3] : 1;
        shape_[1] = shape.size() > 1 ? shape[1] : 1;
        shape_[2] = shape.size() > 2 ? shape[2] : 1;
      break;
      case CNML_NHCW:
        shape_[0] = shape.size() > 0 ? shape[0] : 1;
        shape_[3] = shape.size() > 2 ? shape[2] : 1;
        shape_[1] = shape.size() > 1 ? shape[1] : 1;
        shape_[2] = shape.size() > 3 ? shape[3] : 1;
      break;
      case CNML_NWCH:
        shape_[0] = shape.size() > 0 ? shape[0] : 1;
        shape_[3] = shape.size() > 2 ? shape[2] : 1;
        shape_[1] = shape.size() > 3 ? shape[3] : 1;
        shape_[2] = shape.size() > 1 ? shape[1] : 1;
      break;
      case CNML_NWHC:
        shape_[0] = shape.size() > 0 ? shape[0] : 1;
        shape_[3] = shape.size() > 3 ? shape[3] : 1;
        shape_[1] = shape.size() > 2 ? shape[2] : 1;
        shape_[2] = shape.size() > 1 ? shape[1] : 1;
      break;
      case CNML_CNHW:
        shape_[0] = shape.size() > 1 ? shape[1] : 1;
        shape_[3] = shape.size() > 0 ? shape[0] : 1;
        shape_[1] = shape.size() > 2 ? shape[2] : 1;
        shape_[2] = shape.size() > 3 ? shape[3] : 1;
      break;
      case CNML_CNWH:
        shape_[0] = shape.size() > 1 ? shape[1] : 1;
        shape_[3] = shape.size() > 0 ? shape[0] : 1;
        shape_[1] = shape.size() > 3 ? shape[3] : 1;
        shape_[2] = shape.size() > 2 ? shape[2] : 1;
      break;
      case CNML_CHWN:
        shape_[0] = shape.size() > 3 ? shape[3] : 1;
        shape_[3] = shape.size() > 0 ? shape[0] : 1;
        shape_[1] = shape.size() > 1 ? shape[1] : 1;
        shape_[2] = shape.size() > 2 ? shape[2] : 1;
      break;
      case CNML_CHNW:
        shape_[0] = shape.size() > 2 ? shape[2] : 1;
        shape_[3] = shape.size() > 0 ? shape[0] : 1;
        shape_[1] = shape.size() > 1 ? shape[1] : 1;
        shape_[2] = shape.size() > 3 ? shape[3] : 1;
      break;
      case CNML_CWNH:
        shape_[0] = shape.size() > 2 ? shape[2] : 1;
        shape_[3] = shape.size() > 0 ? shape[0] : 1;
        shape_[1] = shape.size() > 3 ? shape[3] : 1;
        shape_[2] = shape.size() > 1 ? shape[1] : 1;
      break;
      case CNML_CWHN:
        shape_[0] = shape.size() > 3 ? shape[3] : 1;
        shape_[3] = shape.size() > 0 ? shape[0] : 1;
        shape_[1] = shape.size() > 2 ? shape[2] : 1;
        shape_[2] = shape.size() > 1 ? shape[1] : 1;
      break;
      case CNML_HNCW:
        shape_[0] = shape.size() > 1 ? shape[1] : 1;
        shape_[3] = shape.size() > 2 ? shape[2] : 1;
        shape_[1] = shape.size() > 0 ? shape[0] : 1;
        shape_[2] = shape.size() > 3 ? shape[3] : 1;
      break;
      case CNML_HNWC:
        shape_[0] = shape.size() > 1 ? shape[1] : 1;
        shape_[3] = shape.size() > 3 ? shape[3] : 1;
        shape_[1] = shape.size() > 0 ? shape[0] : 1;
        shape_[2] = shape.size() > 2 ? shape[2] : 1;
      break;
      case CNML_HCWN:
        shape_[0] = shape.size() > 3 ? shape[3] : 1;
        shape_[3] = shape.size() > 1 ? shape[1] : 1;
        shape_[1] = shape.size() > 0 ? shape[0] : 1;
        shape_[2] = shape.size() > 2 ? shape[2] : 1;
      break;
      case CNML_HCNW:
        shape_[0] = shape.size() > 2 ? shape[2] : 1;
        shape_[3] = shape.size() > 1 ? shape[1] : 1;
        shape_[1] = shape.size() > 0 ? shape[0] : 1;
        shape_[2] = shape.size() > 3 ? shape[3] : 1;
      break;
      case CNML_HWNC:
        shape_[0] = shape.size() > 2 ? shape[2] : 1;
        shape_[3] = shape.size() > 3 ? shape[3] : 1;
        shape_[1] = shape.size() > 0 ? shape[0] : 1;
        shape_[2] = shape.size() > 1 ? shape[1] : 1;
      break;
      case CNML_HWCN:
        shape_[0] = shape.size() > 3 ? shape[3] : 1;
        shape_[3] = shape.size() > 2 ? shape[2] : 1;
        shape_[1] = shape.size() > 0 ? shape[0] : 1;
        shape_[2] = shape.size() > 1 ? shape[1] : 1;
      break;
      case CNML_WNCH:
        shape_[0] = shape.size() > 1 ? shape[1] : 1;
        shape_[3] = shape.size() > 2 ? shape[2] : 1;
        shape_[1] = shape.size() > 3 ? shape[3] : 1;
        shape_[2] = shape.size() > 0 ? shape[0] : 1;
      break;
      case CNML_WNHC:
        shape_[0] = shape.size() > 1 ? shape[1] : 1;
        shape_[3] = shape.size() > 3 ? shape[3] : 1;
        shape_[1] = shape.size() > 2 ? shape[2] : 1;
        shape_[2] = shape.size() > 0 ? shape[0] : 1;
      break;
      case CNML_WCHN:
        shape_[0] = shape.size() > 3 ? shape[3] : 1;
        shape_[3] = shape.size() > 1 ? shape[1] : 1;
        shape_[1] = shape.size() > 2 ? shape[2] : 1;
        shape_[2] = shape.size() > 0 ? shape[0] : 1;
      break;
      case CNML_WCNH:
        shape_[0] = shape.size() > 2 ? shape[2] : 1;
        shape_[3] = shape.size() > 1 ? shape[1] : 1;
        shape_[1] = shape.size() > 3 ? shape[3] : 1;
        shape_[2] = shape.size() > 0 ? shape[0] : 1;
      break;
      case CNML_WHNC:
        shape_[0] = shape.size() > 2 ? shape[2] : 1;
        shape_[3] = shape.size() > 3 ? shape[3] : 1;
        shape_[1] = shape.size() > 1 ? shape[1] : 1;
        shape_[2] = shape.size() > 0 ? shape[0] : 1;
      break;
      case CNML_WHCN:
        shape_[0] = shape.size() > 3 ? shape[3] : 1;
        shape_[3] = shape.size() > 2 ? shape[2] : 1;
        shape_[1] = shape.size() > 1 ? shape[1] : 1;
        shape_[2] = shape.size() > 0 ? shape[0] : 1;
      break;
      case CNML_ARRAY:
        shape_ = shape;
      break;
      default:
       LOG(FATAL) << "Unsupported mluDataOrder! " << int(shape_order);
      break;
    }
  } else {
    switch (shape_order) {
      case CNML_NCDHW:
        shape_[0] = shape[0];
        shape_[4] = shape[1];
        shape_[1] = shape[2];
        shape_[2] = shape[3];
        shape_[3] = shape[4];
      break;
      case CNML_NDHWC:
        shape_[0] = shape[0];
        shape_[4] = shape[4];
        shape_[1] = shape[1];
        shape_[2] = shape[2];
        shape_[3] = shape[3];
      break;
      case CNML_DHWCN:
        shape_[0] = shape[4];
        shape_[4] = shape[3];
        shape_[1] = shape[0];
        shape_[2] = shape[1];
        shape_[3] = shape[2];
      break;
      case CNML_ARRAY:
        shape_ = shape;
      break;
      default:
        shape_[0] = shape[0];
        shape_[4] = shape[1];
        shape_[1] = shape[2];
        shape_[2] = shape[3];
        shape_[3] = shape[4];
      break;
    }
  }
  dim_ = shape_.size();
}

vector<int> MLUTensorDesc::cpu_shape() const {
  // shape : NC(D)HW --> N(D)HWC
  vector<int> cpu_shape(dim_, 1);
  int channel = shape_[dim_ - 1];
  for (int i = 2; i <shape_.size(); i++) {
    cpu_shape[i] = shape_[i - 1];
  }
  cpu_shape[0] = shape_[0];
  cpu_shape[1] = channel;
  return cpu_shape;
}

void MLUTensorDesc::Create() {
  if (mlu_tensor_ == nullptr) {
    MLU_CHECK(cnmlCreateTensor_V2(&mlu_tensor_, tensor_type_));
    int dim_shape[shape_.size()];
    std::copy(shape_.begin(), shape_.end(), dim_shape);
    int* dim_strides = nullptr;
    vector<int> strides(dim_ , 0);
    if (has_dim_strides_) {
      // strides : NC(D)HW --> N(D)HWC
      strides[0] = dim_strides_[0];
      strides[dim_-1] = dim_strides_[1];
      for (int i = 1; i < dim_-1; i++) {
        strides[i] = dim_strides_[i+1];
      }
      dim_strides = strides.data();
      // mark first conv input tensor
      if (mlu_dtype_ == DT_UINT8 && shape_[3] == 3) {
        is_first_conv_input_tensor_ = true;
      }
    }
    MLU_CHECK(cnmlSetTensorShape_V2(mlu_tensor_, dim_,
          dim_shape, dim_strides));
    MLU_CHECK(cnmlSetTensorDataType(mlu_tensor_, to_cnml_dtype(mlu_dtype_)));
  }
}

void MLUTensorDesc::set_dim_strides(vector<int> dim_strides) {
  dim_strides_ = dim_strides;
  has_dim_strides_ = true;
}

void MLUTensorDesc::Destroy() {
  if (mlu_tensor_ != nullptr) {
    MLU_CHECK(cnmlDestroyTensor(&mlu_tensor_));
    mlu_tensor_ = nullptr;
  }
}

const cnmlTensor_t MLUTensorDesc::mlu() const {
  return mlu_tensor_;
}

void MLUTensorDesc::set_position(int position) {
  position_ = position;
  has_position_ = true;
  Create();
  cnmlSetQuantizedPosition(mlu_tensor_, position_);
}

void MLUTensorDesc::set_scale(float scale) {
  scale_ = scale;
  has_scale_ = true;
  Create();
  cnmlSetQuantizedScale(mlu_tensor_, scale_);
}

void MLUTensorDesc::set_positions(const vector<int>& positions) {
  positions_ = positions;
  has_position_ = true;
  Create();
  cnmlSetQuantizedPositionByChannel(mlu_tensor_, &positions_[0], positions_.size());
}

void MLUTensorDesc::set_scales(const vector<float>& scales) {
  scales_ = scales;
  has_scale_ = true;
  Create();
  cnmlSetQuantizedScaleByChannel(mlu_tensor_, &scales_[0], scales_.size());
}

MLUTensorDesc::~MLUTensorDesc() {
  Destroy();
}

}  // namespace caffe

#endif  // USE_MLU
