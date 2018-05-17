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
#pragma once;

#include "framework/attribute.h"
#include <map>
#include <string>

namespace paddle_mobile {

    namespace framework {
        template <typename Dtype> class OperatorBase;
        class OpDesc;
        class BlockDesc;
        class InferShapeContext;
    }

    using VariableNameMap = std::map<std::string, std::vector<std::string>>;

    template <typename Dtype>
    using OpCreator = std::function<framework::OperatorBase<Dtype> *(
        const std::string & /*type*/, const VariableNameMap & /*inputs*/,
        const VariableNameMap & /*outputs*/,
        const framework::AttributeMap & /*attrs*/)>;

    using GradOpMakerFN =
        std::function<std::vector<std::unique_ptr<framework::OpDesc>>(
            const framework::OpDesc &,
            const std::unordered_set<std::string> & /*no_grad_set*/,
            std::unordered_map<std::string, std::string> * /*grad_to_var*/,
            const std::vector<framework::BlockDesc *> &grad_block)>;

    using InferVarTypeFN =
        std::function<void(const framework::OpDesc & /*op_desc*/,
                           framework::BlockDesc * /*block*/)>;

    using InferShapeFN = std::function<void(framework::InferShapeContext *)>;
};
