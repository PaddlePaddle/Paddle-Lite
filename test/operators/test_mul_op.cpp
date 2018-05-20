
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
#include "../test_include.h"
#include "operators/mul_op.h"

namespace paddle_mobile {
namespace framework {

template <typename Dtype> class TestMulOp {
public:
  explicit TestMulOp(const Program<Dtype> p) : program_(p) {
    if (use_optimize_) {
      to_predict_program_ = program_.optimizeProgram;
    } else {
      to_predict_program_ = program_.originProgram;
    }

    const std::vector<std::shared_ptr<BlockDesc>> blocks =
        to_predict_program_->Blocks();
    //  DLOG << " **block size " << blocks.size();
    for (int i = 0; i < blocks.size(); ++i) {
      std::shared_ptr<BlockDesc> block_desc = blocks[i];
      std::vector<std::shared_ptr<OpDesc>> ops = block_desc->Ops();
      //    DLOG << " ops " << ops.size();
      for (int j = 0; j < ops.size(); ++j) {
        std::shared_ptr<OpDesc> op = ops[j];
        if (op->Type() == "mul" && op->Input("X")[0] == "pool2d_0.tmp_0") {
          DLOG << " mul attr size: " << op->GetAttrMap().size();
          DLOG << " inputs size: " << op->GetInputs().size();
          DLOG << " outputs size: " << op->GetOutputs().size();
          DLOG << " Input X is : " << op->Input("X")[0];
          DLOG << " Input Y is : " << op->Input("Y")[0];
          DLOG << " Output Out is : " << op->Output("Out")[0];
          DLOG << "x_num_col_dims : "
               << op->GetAttrMap().at("x_num_col_dims").Get<int>();
          DLOG << "y_num_col_dims : "
               << op->GetAttrMap().at("y_num_col_dims").Get<int>();

          std::shared_ptr<operators::MulOp<Dtype, float>> mul =
              std::make_shared<operators::MulOp<Dtype, float>>(
                  op->Type(), op->GetInputs(), op->GetOutputs(),
                  op->GetAttrMap(), program_.scope);
          ops_of_block_[*block_desc.get()].push_back(mul);
        }
      }
    }
  }

  std::shared_ptr<Tensor> predict_mul(Tensor &t1, Tensor &t2) {
    // feed
    auto scope = program_.scope;
    Variable *x_feed_value = scope->Var("pool2d_0.tmp_0");
    auto tensor_x = x_feed_value->GetMutable<Tensor>();
    tensor_x->ShareDataWith(t1);

    Variable *y_feed_value = scope->Var("fc_0.w_0");
    auto tensor_y = y_feed_value->GetMutable<Tensor>();
    tensor_y->ShareDataWith(t2);

    Variable *con_output = scope->Var("fc_0.tmp_0");
    auto *output_tensor = con_output->GetMutable<Tensor>();
    output_tensor->mutable_data<float>({3, 3});
    //  DLOG << typeid(output_tensor).name();
    //  DLOG << "output_tensor dims: " << output_tensor->dims();

    std::shared_ptr<Tensor> out_tensor = std::make_shared<LoDTensor>();
    out_tensor.reset(output_tensor);

    predict_mul(t1, t2, 0);
    return out_tensor;
  }

private:
  const framework::Program<Dtype> program_;
  std::shared_ptr<ProgramDesc> to_predict_program_;
  std::map<framework::BlockDesc,
           std::vector<std::shared_ptr<OperatorBase<Dtype>>>>
      ops_of_block_;
  bool use_optimize_ = false;

  void predict_mul(const Tensor &t1, const Tensor &t2, int block_id) {
    std::shared_ptr<BlockDesc> to_predict_block =
        to_predict_program_->Block(block_id);
    for (int j = 0; j < ops_of_block_[*to_predict_block.get()].size(); ++j) {
      auto op = ops_of_block_[*to_predict_block.get()][j];
      DLOG << "op -> run()";
      op->Run();
    }
  }
};

template class TestMulOp<CPU>;
} // namespace framework
} // namespace paddle_mobile

int main() {
  DLOG << "----------**********----------";
  DLOG << "begin to run MulOp Test";
  paddle_mobile::Loader<paddle_mobile::CPU> loader;
  auto program =
      loader.Load(std::string("../../test/models/"
                              "image_classification_resnet.inference.model"));

  /// input x (3,2,1,1)
  paddle_mobile::framework::Tensor inputx;
  SetupTensor<float>(&inputx, {3, 2, 1, 1}, static_cast<float>(0),
                     static_cast<float>(1));
  auto *inputx_ptr = inputx.data<float>();

  /// input y (2,3)
  paddle_mobile::framework::Tensor inputy;
  SetupTensor<float>(&inputy, {2, 3}, static_cast<float>(0),
                     static_cast<float>(1));
  auto *inputy_ptr = inputy.data<float>();

  paddle_mobile::framework::TestMulOp<paddle_mobile::CPU> testMulOp(program);

  auto output_mul = testMulOp.predict_mul(inputx, inputy);
  auto *output_mul_ptr = output_mul->data<float>();

  auto dimx_1 = inputx.numel() / inputx.dims()[0];
  DLOG << " inputx : ";
  for (int i = 0; i < inputx.dims()[0]; ++i) {
    for (int j = 0; j < dimx_1; ++j) {
      DLOGF("%f ", inputx_ptr[i * dimx_1 + j]);
    }
    DLOGF("\n");
  }

  auto dimy_1 = inputy.numel() / inputy.dims()[0];
  DLOG << " inputy : ";
  for (int i = 0; i < inputy.dims()[0]; ++i) {
    for (int j = 0; j < dimy_1; ++j) {
      DLOGF("%f ", inputy_ptr[i * dimx_1 + j]);
    }
    DLOGF("\n");
  }

  auto dim_output_1 = output_mul->numel() / output_mul->dims()[0];
  DLOG << " output : ";
  for (int i = 0; i < output_mul->dims()[0]; ++i) {
    for (int j = 0; j < dim_output_1; ++j) {
      DLOGF("%f ", output_mul_ptr[i * dimy_1 + j]);
    }
    DLOGF("\n");
  }

  /// output (3,3)
  DLOG << "output memory size : " << output_mul->memory_size();
  DLOG << "output numel : " << output_mul->numel();

  DLOG << inputx_ptr[0] << " x " << inputy_ptr[0] << " + " << inputx_ptr[1]
       << " x " << inputy_ptr[0 + 3] << " = " << output_mul_ptr[0];
  return 0;
}
