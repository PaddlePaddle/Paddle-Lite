
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

#include "executor.h"
#include "lod_tensor.h"
#include "operators/conv_op.h"
#include "variable.h"

namespace paddle_mobile {
namespace framework {

template <typename Dtype>
Executor<Dtype>::Executor(const Program<Dtype> p) : program_(p) {
  if (use_optimize_) {
    to_predict_program_ = program_.optimizeProgram;
  } else {
    to_predict_program_ = program_.originProgram;
  }

  const std::vector<std::shared_ptr<BlockDesc>> blocks =
      to_predict_program_->Blocks();
  std::cout << " **block size " << blocks.size() << std::endl;
  for (int i = 0; i < blocks.size(); ++i) {
    std::shared_ptr<BlockDesc> block_desc = blocks[i];
    std::vector<std::shared_ptr<OpDesc>> ops = block_desc->Ops();
    std::cout << " ops " << ops.size() << std::endl;
    for (int j = 0; j < ops.size(); ++j) {
      std::shared_ptr<OpDesc> op = ops[j];
      //        std::cout << " input 0 " << op->Input("Input")[0] << std::endl;
      if (op->Type() == "conv2d" && op->Input("Input")[0] == "pixel") {
        std::cout << " conv2d attr size: " << op->GetAttrMap().size()
                  << std::endl;
        std::cout << " input size: " << op->GetInputs().size() << std::endl;

        std::cout << " output size: " << op->GetOutputs().size() << std::endl;

        Attribute strides_attr = op->GetAttrMap().at("strides");
        std::vector<int> stride = strides_attr.Get<std::vector<int>>();
        for (int k = 0; k < stride.size(); ++k) {
          std::cout << " stride " << stride[k] << std::endl;
        }

        std::shared_ptr<operators::ConvOp<Dtype, float>> conv =
            std::make_shared<operators::ConvOp<Dtype, float>>(
                op->Type(), op->GetInputs(), op->GetOutputs(), op->GetAttrMap(),
                program_.scope);
        ops_of_block_[*block_desc.get()].push_back(conv);
      }
    }
  }
}

template <typename Dtype>
std::shared_ptr<Tensor> Executor<Dtype>::predict(Tensor &t) {
  // feed
  auto scope = program_.scope;
  Variable *g_feed_value = scope->Var("pixel");
  auto tensor = g_feed_value->GetMutable<Tensor>();
  tensor->ShareDataWith(t);

  Variable *con_output = scope->Var("conv2d_0.tmp_0");
  Tensor *output_tensor = con_output->GetMutable<Tensor>();
  output_tensor->mutable_data<float>({1, 16, 32, 32});
  std::cout << typeid(output_tensor).name() << std::endl;
  std::cout << "output_tensor dims: " << output_tensor->dims() << std::endl;

  std::shared_ptr<Tensor> out_tensor = std::make_shared<LoDTensor>();
  out_tensor.reset(output_tensor);

  predict(t, 0);
  return out_tensor;
}

template <typename Dtype>
void Executor<Dtype>::predict(const Tensor &t, int block_id) {
  std::shared_ptr<BlockDesc> to_predict_block =
      to_predict_program_->Block(block_id);
  for (int j = 0; j < ops_of_block_[*to_predict_block.get()].size(); ++j) {
    auto op = ops_of_block_[*to_predict_block.get()][j];
    std::cout << "开始run" << std::endl;
    op->Run();
  }
}

template class Executor<ARM>;

}  // namespace framework
}  // namespace paddle_mobile
