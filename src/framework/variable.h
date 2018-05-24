#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>
#include "paddle_mobile_object.h"

namespace paddle_mobile {
namespace framework {
class Variable : public PaddleMobileObject {
 public:
  template <typename T>
  const T *Get() const {
    return static_cast<const T *>(holder_->Ptr());
  }

  bool IsInitialized() const { return holder_ != nullptr; }

  const std::string *Name() { return name_; }

  template <typename T>
  T *GetMutable() {
    if (!IsType<T>()) {
      if (*Name() == "pixel") {
        //        std::cout << " reset " << *Name() <<
        //        std::endl;
      }
      holder_.reset(new PlaceholderImp<T>(new T()));
    }
    return static_cast<T *>(holder_->Ptr());
  }

  template <typename T>
  bool IsType() const {
    if (holder_) {
      //                printf("not null \n");
      printf(" holder type : %s, this type %s \n", holder_->Type().name(),
             typeid(T).name());
    }

    //              std::cout << " " << holder_->Type() << " " <<
    //              typeid(T) <<
    //              std::endl;
    return holder_ != nullptr && holder_->Type() == typeid(T);
  }

  void Clear() { holder_.reset(); }

  std::type_index Type() const { return holder_->Type(); }

  void SetName(const std::string *name) { name_ = name; }

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

  std::unique_ptr<Placeholder> holder_;
  friend class Scope;
  const std::string *name_;
};
}  // namespace framework
}  // namespace paddle_mobile
