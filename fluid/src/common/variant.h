//
// Created by liuRuiLong on 2018/5/3.
//

#include <iostream>

#pragma once;


namespace paddle_mobile{
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
                reinterpret_cast<F*>(data)->~F();
            }else{
                std::cout << "未匹配到 " << std::endl;
            }
        }
    };

    template<size_t size>
    class RawData {
        char data[size];
    };

    template<typename... Ts>
    struct Variant {
        template<typename T, typename... Args>
        void Set(Args&&... args){
            helper::Destroy(type_id, &data);
            new (&data) T(std::forward<Args>(args)...);
            type_id = typeid(T).hash_code();
        }

        template<typename T>
        T& Get(){
            if (type_id == typeid(T).hash_code())
                return *reinterpret_cast<T*>(&data);
            else
                throw std::bad_cast();
        }

    private:
        static inline size_t invalid_type() {
            return typeid(void).hash_code();
        }
        Variant() : type_id(invalid_type()) {}
        typedef VariantHelper<Ts...> helper;
        size_t type_id;
        RawData<helper::size> data;
        ~Variant() { helper::Destroy(type_id, &data); }
    };

}

