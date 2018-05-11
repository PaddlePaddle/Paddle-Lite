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

#include <iostream>

#pragma once


namespace paddle_mobile{
  template <int ID, typename  Type>
  struct IDToType{
      typedef Type type_t;
  };

  template<typename F, typename... Ts>
  struct VariantHelper {
      static const size_t size = sizeof(F) > VariantHelper<Ts...>::size ? sizeof(F) : VariantHelper<Ts...>::size;

      inline static void Destroy(size_t id, void *data){
          if (id == typeid(F).hash_code()){
              reinterpret_cast<F*>(data)->~F();
          }else{
              VariantHelper<Ts...>::Destroy(id, data);
          }
      }
  };

  template<typename F>
  struct VariantHelper<F>  {
      static const size_t size = sizeof(F);
      inline static void Destroy(size_t id, void * data){
          if (id == typeid(F).hash_code()){
//              reinterpret_cast<F*>(data)->~F();
          }else{
//              std::cout << "未匹配到 " << std::endl;
          }
      }
  };

  template<size_t size>
  class RawData {
  public:
      char data[size];
      RawData(){}
      RawData(const RawData & raw_data){
          strcpy(data, raw_data.data);
      }
//      void operator=(const RawData &raw_data){
//        strcpy(data, raw_data.data);
//      }

  };

  template<typename... Ts>
  struct Variant {
      Variant(const Variant &variant){
//        std::cout << " 赋值构造函数 " << std::endl;
          type_id = variant.type_id;
          data = variant.data;
      }

      Variant() : type_id(invalid_type()) {}
      ~Variant() {
//        helper::Destroy(type_id, &data);
      }
      template<typename T, typename... Args>
      void Set(Args&&... args){
          helper::Destroy(type_id, &data);
          new (&data) T(std::forward<Args>(args)...);
          type_id = typeid(T).hash_code();
      }

      template<typename T>
      T& Get() const {
          if (type_id == typeid(T).hash_code()){
              return *const_cast<T*>(reinterpret_cast<const T*>(&data));
          } else {
              std::cout << " bad cast in variant " << std::endl;
              throw std::bad_cast();
          }
      }

      size_t TypeId() const {
          return type_id;
      }

      size_t TypeId() {
          return type_id;
      }
  private:
      static inline size_t invalid_type() {
          return typeid(void).hash_code();
      }
      typedef VariantHelper<Ts...> helper;
      size_t type_id;
      RawData<helper::size> data;
  };

  template <typename T>
  struct Vistor{
      typedef T type_t;
  };

}

