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

#include "block_desc.h"


namespace paddle_mobile {

namespace framework {
    std::vector<std::shared_ptr<VarDesc>>  BlockDesc::Vars() const{
        std::vector<std::shared_ptr<VarDesc>> res;
        for (const auto &p : vars_) {
            res.push_back(p.second);
        }
        return res;
    }

    std::vector<std::shared_ptr<OpDesc>> BlockDesc::Ops() const{
        std::vector<std::shared_ptr<OpDesc>> res;
        for (const auto &op : ops_) {
            res.push_back(op);
        }
        return res;

    }

    BlockDesc::BlockDesc(const proto::BlockDesc &desc): desc_(desc){
        for(const proto::VarDesc &var_desc : desc_.vars()){
            vars_[var_desc.name()].reset(new VarDesc(var_desc));
        }
        for(const proto::OpDesc &op_desc : desc_.ops()){
            ops_.emplace_back(new framework::OpDesc(op_desc));
        }
    }
}

}