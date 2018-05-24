#include <iostream>

#include "common/log.h"

#pragma once

namespace paddle_mobile {
template <int ID, typename Type>
struct IDToType {
  typedef Type type_t;
};

template <typename F, typename... Ts>
struct VariantHelper {
  static const size_t size = sizeof(F) > VariantHelper<Ts...>::size
                                 ? sizeof(F)
                                 : VariantHelper<Ts...>::size;

  inline static void Destroy(size_t id, void *data) {
    if (id == typeid(F).hash_code()) {
      reinterpret_cast<F *>(data)->~F();
    } else {
      VariantHelper<Ts...>::Destroy(id, data);
    }
  }
};

template <typename F>
struct VariantHelper<F> {
  static const size_t size = sizeof(F);
  inline static void Destroy(size_t id, void *data) {
    if (id == typeid(F).hash_code()) {
      //              reinterpret_cast<F*>(data)->~F();
    } else {
      //              std::cout << "未匹配到 " << std::endl;
    }
  }
};

template <size_t size>
class RawData {
 public:
  char data[size];
  RawData() {}
  RawData(const RawData &raw_data) { strcpy(data, raw_data.data); }
  //      void operator=(const RawData &raw_data){
  //        strcpy(data, raw_data.data);
  //      }
};

template <typename... Ts>
struct Variant {
  Variant(const Variant &variant) {
    //        std::cout << " 赋值构造函数 " << std::endl;
    type_id = variant.type_id;
    data = variant.data;
  }

  Variant() : type_id(invalid_type()) {}
  ~Variant() {
    //        helper::Destroy(type_id, &data);
  }

  template <typename T, typename... Args>
  void Set(Args &&... args) {
    helper::Destroy(type_id, &data);
    new (&data) T(std::forward<Args>(args)...);
    type_id = typeid(T).hash_code();
  }

  template <typename T>
  T &Get() const {
    if (type_id == typeid(T).hash_code()) {
      return *const_cast<T *>(reinterpret_cast<const T *>(&data));
    } else {
      //      std::cout << " bad cast in variant " << std::endl;
      throw std::bad_cast();
    }
  }

  size_t TypeId() const { return type_id; }

 private:
  static inline size_t invalid_type() { return typeid(void).hash_code(); }
  typedef VariantHelper<Ts...> helper;
  size_t type_id;
  RawData<helper::size> data;
};

template <typename T>
struct Vistor {
  typedef T type_t;
};

}  // namespace paddle_mobile
