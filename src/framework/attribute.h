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

#include "common/variant.h"
#include "framework.pb.h"

namespace paddle_mobile {
    namespace framework {

        class BlockDesc;

        class Attribute {
          public:
            static Attribute
            GetAttrValue(const proto::OpDesc::Attr &attr_desc) {
                //    std::cout << "begin get attr value" << std::endl;
                Attribute attr;
                switch (attr_desc.type()) {
                case proto::AttrType::BOOLEAN: {
                    attr.Set<bool>(attr_desc.b());
                    break;
                }
                case proto::AttrType::INT: {
                    attr.Set<int>(attr_desc.i());
                    break;
                }
                case proto::AttrType::FLOAT: {
                    attr.Set<float>(attr_desc.f());
                    break;
                }
                case proto::AttrType::STRING: {
                    attr.Set<std::string>(attr_desc.s());
                    break;
                }
                case proto::AttrType::BOOLEANS: {
                    std::vector<bool> val(attr_desc.bools_size());
                    for (int i = 0; i < attr_desc.bools_size(); ++i) {
                        val[i] = attr_desc.bools(i);
                    }
                    attr.Set<std::vector<bool>>(val);
                    break;
                }
                case proto::AttrType::INTS: {
                    std::vector<int> val(attr_desc.ints_size());
                    for (int i = 0; i < attr_desc.ints_size(); ++i) {
                        val[i] = attr_desc.ints(i);
                    }
                    attr.Set<std::vector<int>>(val);
                    break;
                }
                case proto::AttrType::FLOATS: {
                    std::vector<float> val(attr_desc.floats_size());
                    for (int i = 0; i < attr_desc.floats_size(); ++i) {
                        val[i] = attr_desc.floats(i);
                    }
                    attr.Set<std::vector<float>>(val);
                    break;
                }
                case proto::AttrType::STRINGS: {
                    std::vector<std::string> val(attr_desc.strings_size());
                    for (int i = 0; i < attr_desc.strings_size(); ++i) {
                        val[i] = attr_desc.strings(i);
                    }
                    attr.Set<std::vector<std::string>>(val);
                    break;
                }
                case proto::AttrType::LONG: {
                    attr.Set<int64_t>(attr_desc.l());
                    break;
                }
                default:
                    //        std::cout << " not support " << std::endl;
                    break;
                }
                //    std::cout << "end get attr value" << std::endl;
                return attr;
            }

            Attribute() {}
            template <typename T, typename... Args>
            Attribute &Set(Args &&... args) {
                variant_.Set<T>(args...);
                return *this;
            }

            template <typename T> T &Get() const { return variant_.Get<T>(); }

          private:
            Variant<int, float, std::string, std::vector<int>,
                    std::vector<float>, std::vector<std::string>, bool,
                    std::vector<bool>, BlockDesc *, int64_t>
                variant_;
        };

        using AttributeMap = std::unordered_map<std::string, Attribute>;

        class AttrReader {
          public:
            explicit AttrReader(const AttributeMap &attrs) : attrs_(attrs) {}

            template <typename T> inline T Get(const std::string &name) const {
                //          PADDLE_ENFORCE(attrs_.count(name) != 0, "%s should
                //          be in
                //          AttributeMap",
                //                         name);
                return ((Attribute)attrs_.at(name)).Get<T>();
            }

          private:
            const AttributeMap &attrs_;
        };

    } // namespace framework
} // namespace paddle_mobile
