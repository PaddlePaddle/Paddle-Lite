
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

#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>

namespace paddle_mobile {
    namespace framework {

        class Variable {
        public:
            template <typename T>
            const T& Get() const {
                return *static_cast<const T*>(holder_->Ptr());
            }

            bool IsInitialized() const { return holder_ != nullptr; }

            template <typename T>
            T* GetMutable() {
                if (!IsType<T>()) {
                    holder_.reset(new PlaceholderImpl<T>(new T()));
                }
                return static_cast<T*>(holder_->Ptr());
            }

            template <typename T>
            bool IsType() const {
                return holder_ != nullptr &&
                       std::type_index(typeid(T)) == std::type_index(holder_->Type());
            }

            void Clear() { holder_.reset(); }

            std::type_index Type() const {
                return holder_->Type();
            }

        private:
            struct Placeholder {
                virtual ~Placeholder() {}
                virtual const std::type_info& Type() const = 0;
                virtual void* Ptr() const = 0;
            };

            // Placeholder hides type T, so it doesn't appear as a template
            // parameter of Variable.
            template <typename T>
            struct PlaceholderImpl : public Placeholder {
                explicit PlaceholderImpl(T* ptr) : ptr_(ptr), type_(typeid(T)) {}

                virtual const std::type_info& Type() const { return type_; }
                virtual void* Ptr() const { return static_cast<void*>(ptr_.get()); }

                std::unique_ptr<T> ptr_;
                const std::type_info& type_;
            };

            std::unique_ptr<Placeholder>
                    holder_;  // pointers to a PlaceholderImpl object indeed.

            // name_ is only meaningful with a Scope and accessible by it.
            //
            // NOTE: Please don't expose name_ by adding methods like
            // Variable::Name or Scope::VarName!  A variable could have a human
            // readable name or an auto-generated scope-unique name.  In the
            // former case, the caller knows the name and doesn't need to access
            // the name; in the latter case, the variable should be identified
            // by its address but not the unreadable name.
            friend class Scope;
            const std::string* name_;
        };

    }  // namespace framework
}  // namespace paddle




