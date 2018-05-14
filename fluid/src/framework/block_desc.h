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

#include "op_desc.h"
#include "var_desc.h"
#include "framework.pb.h"
#include "paddle_mobile_object.h"

namespace paddle_mobile {
namespace framework {

class BlockDesc: PaddleMobileObject{
public:
  BlockDesc(const proto::BlockDesc &desc);

  const int &ID() const{
    return desc_.idx();
  }

  const int &Parent() const{
    return desc_.parent_idx();
  }

  bool operator==(const paddle_mobile::framework::BlockDesc &in_block) const {
    return this->ID() == in_block.ID() && this->Parent() == in_block.Parent();
  }

  bool operator<(const paddle_mobile::framework::BlockDesc &in_block) const {
    return this->ID() < in_block.ID() && this->Parent() < in_block.Parent();
  }

  std::vector<std::shared_ptr<VarDesc>> Vars() const;
  std::vector<std::shared_ptr<OpDesc>> Ops() const;

private:
  proto::BlockDesc desc_;
  std::vector<std::shared_ptr<OpDesc>> ops_;
  std::unordered_map<std::string, std::shared_ptr<VarDesc>> vars_;
};

}
}

namespace std {

template<> struct hash<paddle_mobile::framework::BlockDesc>
{
  typedef paddle_mobile::framework::BlockDesc argument_type;
  typedef std::size_t result_type;
  result_type operator()(argument_type const& s) const noexcept
  {
    result_type const h1 ( std::hash<int>{}(s.ID()) );
    result_type const h2 ( std::hash<int>{}(s.ID()) );
    return h1 ^ (h2 << 1);
  }
};

}