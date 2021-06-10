#pragma once

#include <cstddef>
#include <memory>
#include "pscore/common/type_util.h"

namespace pscore {

class AnyPtr {
 public:
  AnyPtr() : type_id_(FastTypeId<void>()), ptr_(nullptr) {}
  AnyPtr(std::nullptr_t) : AnyPtr() {}

  template <typename T>
  AnyPtr(T* ptr)
      : type_id_(FastTypeId<T>()),
        ptr_(const_cast<void*>(reinterpret_cast<const void*>(ptr))) {}

  template <typename T>
  T* get() const {
    if (type_id_ != FastTypeId<T>()) {
      return nullptr;
    }
    return reinterpret_cast<T*>(ptr_);
  }

 private:
  void* ptr_{};
  size_t type_id_;
};

}  // namespace pscore
