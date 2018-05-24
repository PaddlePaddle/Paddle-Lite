#pragma once

#include "framework/framework.pb.h"
#include "framework/paddle_mobile_object.h"

namespace paddle_mobile {
namespace framework {

class VarDesc {
 public:
  VarDesc(const proto::VarDesc &desc);

  VarDesc(const VarDesc &var_desc):desc_(var_desc.desc_) {}

  std::string Name() const { return desc_.name(); }

  proto::VarType::Type GetType() const { return desc_.type().type(); }

  bool Persistable() const { return desc_.persistable(); }

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
  std::vector<T> RepeatedToVector(
      const google::protobuf::RepeatedField<T> &repeated_field) const {
    std::vector<T> ret;
    ret.reserve(repeated_field.size());
    std::copy(repeated_field.begin(), repeated_field.end(),
              std::back_inserter(ret));
    return ret;
  }

  std::vector<int64_t> GetShape() const {
    return this->RepeatedToVector(tensor_desc().dims());
  }

 private:
  proto::VarDesc desc_;
};

}  // namespace framework
}  // namespace paddle_mobile
