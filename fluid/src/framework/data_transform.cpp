/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

#pragma once

#include "data_transform.h"

namespace paddle_mobile {
namespace framework {

static void PassTensorData(Tensor* from, Tensor* to) {
  to->ShareDataWith(*from);
  *from = Tensor();
}

void DataTransform(const OpKernelType& expected_kernel_type,
                   const OpKernelType& kernel_type_for_var,
                   const Tensor& input_tensor, Tensor* output_tensor) {
  bool transformed = false;
  Tensor in;
  in.ShareDataWith(input_tensor);
  Tensor out;

//  // do layout transform
//  if (NeedTransformLayout(expected_kernel_type.data_layout_,
//                          kernel_type_for_var.data_layout_)) {
//    TransDataLayout(kernel_type_for_var, expected_kernel_type, in, &out);
//    transformed = true;
//    PassTensorData(&out, &in);
//  }
//
//  // do data type transform
//  if (expected_kernel_type.data_type_ != kernel_type_for_var.data_type_) {
//    TransDataType(kernel_type_for_var, expected_kernel_type, in, &out);
//    transformed = true;
//    PassTensorData(&out, &in);
//  }
//
//  // do device transform
//  if (!platform::is_same_place(kernel_type_for_var.place_,
//                               expected_kernel_type.place_)) {
//    TransDataDevice(in, expected_kernel_type.place_, &out);
//    transformed = true;
//    PassTensorData(&out, &in);
//  }
//
//  PADDLE_ENFORCE(transformed, "No transform is applied, please check!");
  // get output data
  output_tensor->ShareDataWith(in);
}

void CopyVariableWithTensor(const Variable& in_var, const Tensor& tensor,
                            Variable& out_var) {
//  if (in_var.IsType<LoDTensor>()) {
//    auto& in_lod_tensor = in_var.Get<LoDTensor>();
//    auto* tran_lod_tensor = out_var.GetMutable<LoDTensor>();
//    tran_lod_tensor->set_lod(in_lod_tensor.lod());
//    tran_lod_tensor->set_layout(in_lod_tensor.layout());
//    tran_lod_tensor->ShareDataWith(tensor);
//  } else if (in_var.IsType<SelectedRows>()) {
//    auto& in_selected_rows = in_var.Get<SelectedRows>();
//    auto* trans_selected_rows = out_var.GetMutable<SelectedRows>();
//    trans_selected_rows->set_height(in_selected_rows.height());
//    trans_selected_rows->set_rows(in_selected_rows.rows());
//    trans_selected_rows->mutable_value()->ShareDataWith(tensor);
//  } else {
//    PADDLE_THROW("unknown var type");
//  }
}

}  // namespace framework
}  // namespace paddle