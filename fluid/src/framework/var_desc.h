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

#include "framework.pb.h"
#include "paddle_mobile_object.h"

namespace paddle_mobile {
namespace framework {

class VarDesc{
public:
  VarDesc(const proto::VarDesc &desc);

  std::string Name() const {
    return desc_.name();
  }

  proto::VarType::Type GetType() const {
    return desc_.type().type();
  }

  bool Persistable() const {
    return desc_.persistable();
  }

  const proto::VarType::ChannelDesc &channel_desc() const {
    switch (desc_.type().type()) {
      case proto::VarType::CHANNEL:
        return desc_.type().channel();
      default:
        break;
    }
  }

  const proto::VarType::TensorDesc &tensor_desc() const {
    switch (desc_.type().type()) {
      case proto::VarType::SELECTED_ROWS:
        return desc_.type().selected_rows();
      case proto::VarType::LOD_TENSOR:
        return desc_.type().lod_tensor().tensor();
      case proto::VarType::LOD_TENSOR_ARRAY:
        return desc_.type().tensor_array().tensor();
      default:
        break;
    }
  }

  proto::VarType::Type GetDataType() const {
    switch (desc_.type().type()) {
      case proto::VarType::CHANNEL:
        return channel_desc().data_type();
        break;
      default:
        return tensor_desc().data_type();
    }
  }

  template <typename T>
  std::vector<T> RepeatedToVector(const google::protobuf::RepeatedField<T> &repeated_field) const {
    std::vector<T> ret;
    ret.reserve(repeated_field.size());
    std::copy(repeated_field.begin(), repeated_field.end(), std::back_inserter(ret));
    return ret;
  }

  std::vector<int64_t> GetShape() const {
    return this->RepeatedToVector(tensor_desc().dims());
  }

private:
    proto::VarDesc desc_;
};

}
}


