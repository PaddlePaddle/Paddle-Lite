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


#include "op_info.h"
#include "operator.h"
#include "var_type.h"
#include "selected_rows.h"
#include "data_transform.h"
#include "operators/conv_op.h"

namespace paddle_mobile {
namespace framework {

template <typename Dtype>
OperatorBase<Dtype>::OperatorBase(const std::string& type,
                           const VariableNameMap& inputs,
                           const VariableNameMap& outputs,
                           const AttributeMap& attrs,  std::shared_ptr<Scope> scope)
        : type_(type), inputs_(inputs), outputs_(outputs), attrs_(attrs), scope_(scope){
    CheckAllInputOutputSet();
}

template <typename Dtype>
void OperatorBase<Dtype>::Run() {
  RunImpl();
}

template <typename Dtype>
void OperatorBase<Dtype>::CheckAllInputOutputSet() const {
}

template class OperatorBase<ARM>;
template class OperatorWithKernel<ARM>;

} // framework
} // paddle_mobile
