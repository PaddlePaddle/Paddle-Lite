#pragma once
#include "framework.pb.h"
#include "lod_tensor.h"
#include "selected_rows.h"
#include "variable.h"

namespace paddle_mobile {
namespace framework {
inline proto::VarType::Type ToVarType(std::type_index type) {
  if (type.hash_code() == typeid(LoDTensor).hash_code()) {
    return proto::VarType_Type_LOD_TENSOR;
  } else if (type.hash_code() == typeid(SelectedRows).hash_code()) {
    return proto::VarType_Type_SELECTED_ROWS;
  } else {
    //    PADDLE_THROW("ToVarType:Unsupported type %s",
    //    type.name());
  }
}

}  // namespace framework
}  // namespace paddle_mobile
