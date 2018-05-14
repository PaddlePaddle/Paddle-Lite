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

#include "op_param.h"

namespace paddle_mobile {
namespace operators {

std::ostream& operator<<(std::ostream& os, const ConvParam& conv_param) {
  os << "parameter of conv: " << std::endl;
  os << "  stride: "
     << " (" << conv_param.Strides()[0] << conv_param.Strides()[1] << ") "
     << std::endl;
  os << "  paddings: "
     << " (" << conv_param.Paddings()[0] << conv_param.Paddings()[1] << ") "
     << std::endl;
  os << "  dilations: "
     << " (" << conv_param.Dilations()[0] << conv_param.Dilations()[1] << ") "
     << std::endl;
  os << "  groups: " << conv_param.Groups() << std::endl;
  os << "  input  dims: " << conv_param.Input()->dims() << std::endl;
  os << "  filter dims: " << conv_param.Filter()->dims() << std::endl;
  os << "  output dims: " << conv_param.Output()->dims() << std::endl;
  return os;
}

}  // namespace operators
}  // namespace paddle_mobile

