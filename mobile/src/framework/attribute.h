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

#include <cstdlib>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "common/enforce.h"
#include "common/log.h"
#include "common/variant.h"
#include "framework/framework.pb-c.h"

namespace paddle_mobile {
namespace framework {
using std::string;
using std::vector;

class BlockDesc;

class Attribute {
 public:
  static Attribute GetAttrValue(
      PaddleMobile__Framework__Proto__OpDesc__Attr *attr_desc) {
    Attribute attr;
    switch (attr_desc->type) {
      case PADDLE_MOBILE__FRAMEWORK__PROTO__ATTR_TYPE__BOOLEAN: {
        attr.Set<bool>(attr_desc->b);
        break;
      }
      case PADDLE_MOBILE__FRAMEWORK__PROTO__ATTR_TYPE__INT: {
        attr.Set<int>(attr_desc->i);
        break;
      }
      case PADDLE_MOBILE__FRAMEWORK__PROTO__ATTR_TYPE__FLOAT: {
        attr.Set<float>(attr_desc->f);
        break;
      }
      case PADDLE_MOBILE__FRAMEWORK__PROTO__ATTR_TYPE__STRING: {
        attr.Set<std::string>(attr_desc->s);
        break;
      }
      case PADDLE_MOBILE__FRAMEWORK__PROTO__ATTR_TYPE__BOOLEANS: {
        vector<bool> val(attr_desc->n_bools);
        for (int i = 0; i < attr_desc->n_bools; ++i) {
          val[i] = attr_desc->bools[i];
        }
        attr.Set<vector<bool>>(val);
        break;
      }
      case PADDLE_MOBILE__FRAMEWORK__PROTO__ATTR_TYPE__INTS: {
        vector<int> val(attr_desc->n_ints);
        for (int i = 0; i < attr_desc->n_ints; ++i) {
          val[i] = attr_desc->ints[i];
        }
        attr.Set<vector<int>>(val);
        break;
      }
      case PADDLE_MOBILE__FRAMEWORK__PROTO__ATTR_TYPE__FLOATS: {
        vector<float> val(attr_desc->n_floats);
        for (int i = 0; i < attr_desc->n_floats; ++i) {
          val[i] = attr_desc->floats[i];
        }
        attr.Set<vector<float>>(val);
        break;
      }
      case PADDLE_MOBILE__FRAMEWORK__PROTO__ATTR_TYPE__STRINGS: {
        vector<string> val(attr_desc->n_strings);
        for (int i = 0; i < attr_desc->n_strings; ++i) {
          val[i] = attr_desc->strings[i];
        }
        attr.Set<vector<string>>(val);
        break;
      }
      case PADDLE_MOBILE__FRAMEWORK__PROTO__ATTR_TYPE__LONG: {
        attr.Set<int64_t>(attr_desc->l);
        break;
      }
      case PADDLE_MOBILE__FRAMEWORK__PROTO__ATTR_TYPE__BLOCK: {
        break;
      }
      case PADDLE_MOBILE__FRAMEWORK__PROTO__ATTR_TYPE__LONGS: {
        vector<int> val(attr_desc->n_longs);
        for (int i = 0; i < attr_desc->n_longs; ++i) {
          val[i] = attr_desc->longs[i];
        }
        attr.Set<vector<int>>(val);
        break;
      }
      default:
        PADDLE_MOBILE_THROW_EXCEPTION("attr type not support");
    }
    return attr;
  }

  Attribute() {}
  template <typename T, typename... Args>
  Attribute &Set(Args &&... args) {
    variant_.Set<T>(args...);
    return *this;
  }

  template <typename T>
  T &Get() const {
    return variant_.Get<T>();
  }

  std::string GetString() const { return variant_.Get<std::string>(); }

  template <typename Vistor>
  static typename Vistor::type_t ApplyVistor(Vistor vistor, Attribute attr) {
    if (attr.variant_.TypeId() == type_id<int>()) {  // NOLINT
      return vistor(attr.variant_.Get<int>());
    } else if (attr.variant_.TypeId() == type_id<float>()) {  // NOLINT
      return vistor(attr.variant_.Get<float>());
    } else if (attr.variant_.TypeId() == type_id<string>()) {
      return vistor(attr.variant_.Get<std::string>());
    } else if (attr.variant_.TypeId() == type_id<vector<int>>()) {
      return vistor(attr.variant_.Get<vector<int>>());
    } else if (attr.variant_.TypeId() == type_id<vector<float>>()) {
      return vistor(attr.variant_.Get<vector<float>>());
    } else if (attr.variant_.TypeId() == type_id<vector<string>>()) {
      return vistor(attr.variant_.Get<vector<string>>());
    } else if (attr.variant_.TypeId() == type_id<bool>()) {  // NOLINT
      return vistor(attr.variant_.Get<bool>());
    } else if (attr.variant_.TypeId() == type_id<vector<bool>>()) {
      return vistor(attr.variant_.Get<vector<bool>>());
    } else if (attr.variant_.TypeId() == type_id<int64_t>()) {
      return vistor(attr.variant_.Get<int64_t>());
    } else if (attr.variant_.TypeId() == type_id<framework::BlockDesc *>()) {
      return vistor(attr.variant_.Get<framework::BlockDesc *>());
    } else if (attr.variant_.TypeId() ==
               type_id<vector<framework::BlockDesc *>>()) {
      return vistor(attr.variant_.Get<vector<framework::BlockDesc *>>());
    } else if (attr.variant_.TypeId() == type_id<vector<int64_t>>()) {
      return vistor(attr.variant_.Get<vector<int64_t>>());
    } else {
      PADDLE_MOBILE_THROW_EXCEPTION("type not support");
    }
  }

 private:
  Variant<int, float, string, vector<int>, vector<float>, vector<string>, bool,
          vector<bool>, BlockDesc *, vector<BlockDesc *>, int64_t,
          vector<int64_t>>
      variant_;
};

using AttributeMap = std::unordered_map<string, Attribute>;

class AttrReader {
 public:
  explicit AttrReader(const AttributeMap &attrs) : attrs_(attrs) {}

  template <typename T>
  inline T Get(const string &name) const {
    PADDLE_MOBILE_ENFORCE(attrs_.count(name) != 0,
                          "%s should  be in AttributeMap", name.c_str());
    return ((Attribute)attrs_.at(name)).Get<T>();
  }

 private:
  const AttributeMap &attrs_;
};

Print &operator<<(Print &printer, const Attribute &op_desc);

}  // namespace framework
}  // namespace paddle_mobile
