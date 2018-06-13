/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>
#include "../common/variant.h"

namespace paddle_mobile {
namespace framework {
using std::string;

class Variable {
 public:
  template <typename T>
  const T *Get() const {
    return static_cast<const T *>(holder_->Ptr());
  }

  template <typename T>
  const T GetValue() const {
    return variant.Get<T>();
  }

  template <typename T>
  void SetValue(T value) {
    variant.Set<T>(value);
  }

  bool IsInitialized() const { return holder_ != nullptr; }

  template <typename T>
  T *GetMutable() {
    if (!IsType<T>()) {
      holder_.reset(new PlaceholderImp<T>(new T()));
    }
    return static_cast<T *>(holder_->Ptr());
  }

  template <typename T>
  bool IsType() const {
    return holder_ != nullptr && holder_->Type() == typeid(T);
  }

  void Clear() { holder_.reset(); }

  std::type_index Type() const { return holder_->Type(); }

 private:
  struct Placeholder {
    Placeholder() = default;
    virtual ~Placeholder() = default;

    virtual const std::type_info &Type() const = 0;
    virtual void *Ptr() const = 0;
  };

  template <typename T>
  struct PlaceholderImp : public Placeholder {
    explicit PlaceholderImp(T *ptr) : ptr_(ptr), type_(typeid(T)) {}

    virtual const std::type_info &Type() const { return type_; }
    virtual void *Ptr() const override {
      return static_cast<void *>(ptr_.get());
    }

    std::unique_ptr<T> ptr_;
    const std::type_info &type_;
  };
  Variant<int, bool, string, float, double> variant;
  std::unique_ptr<Placeholder> holder_;
  friend class Scope;
  string name_;
};
}  // namespace framework
}  // namespace paddle_mobile
