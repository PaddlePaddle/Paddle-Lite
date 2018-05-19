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

#include "executor_for_test.h"

template <typename DeviceType, typename OpType>
Executor4Test<DeviceType, OpType>::Executor4Test(const Program<DeviceType> p,
                                                 std::string op_type)
    : Executor<DeviceType>(p) {

    if (this->program_.originProgram == nullptr) {
        LOG(paddle_mobile::LogLevel::kLOG_ERROR)
            << "to_predict_program_ == nullptr";
    }

    const std::vector<std::shared_ptr<BlockDesc>> blocks =
        this->to_predict_program_->Blocks();

    for (int i = 0; i < blocks.size(); ++i) {
        std::shared_ptr<BlockDesc> block_desc = blocks[i];
        std::vector<std::shared_ptr<OpDesc>> ops = block_desc->Ops();
        for (int j = 0; j < ops.size(); ++j) {
            std::shared_ptr<OpDesc> op = ops[j];
            if (op->Type() == op_type) {
                std::shared_ptr<OpType> op_ptr = std::make_shared<OpType>(
                    op->Type(), op->GetInputs(), op->GetOutputs(),
                    op->GetAttrMap(), this->program_.scope);

                this->ops_of_block_[*block_desc.get()].push_back(op_ptr);
                break;
            }
        }
    }
}

template <typename DeviceType, typename OpType>
std::shared_ptr<Tensor>
Executor4Test<DeviceType, OpType>::predict(Tensor &t, std::string input,
                                           std::string output, DDim dDim) {

    auto scope = this->program_.scope;
    Variable *g_feed_value = scope->Var(input);
    auto tensor = g_feed_value->GetMutable<Tensor>();
    tensor->ShareDataWith(t);

    Variable *con_output = scope->Var(output);
    Tensor *output_tensor = con_output->GetMutable<Tensor>();
    output_tensor->mutable_data<float>(dDim);
    std::shared_ptr<Tensor> out_tensor = std::make_shared<LoDTensor>();
    out_tensor.reset(output_tensor);

    Executor<DeviceType>::predict(t, 0);
    return out_tensor;
}

template class Executor4Test<
    paddle_mobile::CPU,
    paddle_mobile::operators::ConvOp<paddle_mobile::CPU, float>>;
template class Executor4Test<
    paddle_mobile::CPU,
    paddle_mobile::operators::PoolOp<paddle_mobile::CPU, float>>;
