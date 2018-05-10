
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
#include "variable.h"
#include "lod_tensor.h"
#include "operators/conv_op.h"


using std::cout;
namespace paddle_mobile {

namespace framework {

template <typename Dtype>
Executor<Dtype>::Executor() {}

template <typename Dtype>
Executor<Dtype>::Executor(const Program<Dtype> p): program_(p){
  if (use_optimize_) {
  } else {
    const std::vector<std::shared_ptr<BlockDesc> > blocks = program_.originProgram->Blocks();
    for (int i = 0; i < blocks.size(); ++i) {
      std::shared_ptr<BlockDesc> block_desc = blocks[i];
      std::vector<std::shared_ptr<OpDesc> > ops = block_desc->Ops();
      for (int j = 0; j < ops.size(); ++j) {
        std::shared_ptr<OpDesc> op = ops[j];
        if (op->Type() == "conv2d") {
          operators::ConvOp<Dtype, int> conv(op->Type(), op->GetInputs(), op->GetOutputs(), op->GetAttrMap());
          ops_of_block_[*block_desc.get()].push_back(std::make_shared<operators::ConvOp<Dtype, int> >(conv));
        }
      }
    }
  }
}

template <typename Dtype>
std::shared_ptr<Tensor> Executor<Dtype>::predict(Tensor &t){

  // feed
  auto scope = program_.scope;
  Variable* g_feed_value = scope->Var("pixel");
  auto tensor = g_feed_value->GetMutable<LoDTensor>();
//  tensor->ShareDataWith(t);

  Variable *con_output = scope->Var("conv2d_0.tmp_0");


  LoDTensor *output_tensor = con_output->GetMutable<LoDTensor>();

  std::shared_ptr<Tensor> out_tensor = std::make_shared<LoDTensor>();
  out_tensor.reset(output_tensor);


  std::vector<int> ddims{1, 16, 32, 32};
  DDim ddim = make_ddim(ddims);
  output_tensor->mutable_data<float>(ddim);


  if (use_optimize_) {
  }else{
    for (int i = 0; i < program_.originProgram->Blocks().size(); ++i) {
      auto block = program_.originProgram->Blocks()[i];
      for (int j = 0; j < ops_of_block_[*block.get()].size(); ++j) {
        auto op = ops_of_block_[*block.get()][j];
        op->Run(*(program_.scope.get()));
      }
    }
  }

  return out_tensor;
}

template class Executor<ARM>;


}

}  // namespace paddle



