
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
#include "operators/lrn_op.h"

namespace paddle_mobile {
namespace framework {

template <typename Dtype> class TestLrnOp {
  public:
    explicit TestLrnOp(const Program<Dtype> p) : program_(p) {
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
                //                        if (op->Type() == "mul") {
                //                            DLOG << "x_num_col_dims :
                //                            "
                //                                 << op->GetAttrMap()
                //                                        .at("x_num_col_dims")
                //                                        .Get<int>();
                //                            DLOG << "y_num_col_dims :
                //                            "
                //                                 << op->GetAttrMap()
                //                                        .at("y_num_col_dims")
                //                                        .Get<int>();
                //                            DLOG << " Input X is : "
                //                            << op->Input("X")[0];
                //                        }
                //                        DLOG << "op:" << op->Type();
                if (op->Type() == "lrn" &&
                    op->Input("X")[0] == "pool2d_0.tmp_0") {
                    DLOG << " mul attr size: " << op->GetAttrMap().size();
                    DLOG << " inputs size: " << op->GetInputs().size();
                    DLOG << " outputs size: " << op->GetOutputs().size();
                    DLOG << " Input X is : " << op->Input("X")[0];
                    DLOG << " Output Out is : " << op->Output("Out")[0];
                    DLOG << " n : " << op->GetAttrMap().at("n").Get<int>();
                    DLOG << " alpha : "
                         << op->GetAttrMap().at("alpha").Get<float>();
                    DLOG << " beta : "
                         << op->GetAttrMap().at("beta").Get<float>();
                    DLOG << " k : " << op->GetAttrMap().at("k").Get<float>();
                    std::shared_ptr<operators::LrnOp<Dtype, float>> lrn =
                        std::make_shared<operators::LrnOp<Dtype, float>>(
                            op->Type(), op->GetInputs(), op->GetOutputs(),
                            op->GetAttrMap(), program_.scope);
                    ops_of_block_[*block_desc.get()].push_back(lrn);
                }
            }
        }
    }

    std::shared_ptr<Tensor> predict_lrn(Tensor &t1) {
        // feed
        auto scope = program_.scope;
        Variable *x1_feed_value = scope->Var("pool2d_0.tmp_0");
        auto tensor_x1 = x1_feed_value->GetMutable<Tensor>();
        tensor_x1->ShareDataWith(t1);

        Variable *con_output = scope->Var("pool1_norm1.tmp_1");
        auto *output_tensor = con_output->GetMutable<Tensor>();
        output_tensor->mutable_data<float>({3, 4, 2, 2});
        //  DLOG << typeid(output_tensor).name();
        //  DLOG << "output_tensor dims: " << output_tensor->dims();

        std::shared_ptr<Tensor> out_tensor = std::make_shared<LoDTensor>();
        out_tensor.reset(output_tensor);

        predict_lrn(t1, 0);
        return out_tensor;
    }

  private:
    const framework::Program<Dtype> program_;
    std::shared_ptr<ProgramDesc> to_predict_program_;
    std::map<framework::BlockDesc,
             std::vector<std::shared_ptr<OperatorBase<Dtype>>>>
        ops_of_block_;
    bool use_optimize_ = false;

    void predict_lrn(const Tensor &t1, int block_id) {
        std::shared_ptr<BlockDesc> to_predict_block =
            to_predict_program_->Block(block_id);
        for (int j = 0; j < ops_of_block_[*to_predict_block.get()].size();
             ++j) {
            auto op = ops_of_block_[*to_predict_block.get()][j];
            DLOG << "op -> run()";
            op->Run();
        }
    }
};

template class TestLrnOp<CPU>;
} // namespace framework
} // namespace paddle_mobile

int main() {
    DLOG << "----------**********----------";
    DLOG << "begin to run LrnOp Test";
    paddle_mobile::Loader<paddle_mobile::CPU> loader;
    auto program = loader.Load(std::string("../../test/models/googlenet"));

    /// input x (3,4,2,2)
    paddle_mobile::framework::Tensor inputx1;
    SetupTensor<float>(&inputx1, {3, 4, 2, 2}, static_cast<float>(0),
                       static_cast<float>(1));
    auto *inputx1_ptr = inputx1.data<float>();

    paddle_mobile::framework::TestLrnOp<paddle_mobile::CPU> testLrnOp(program);

    auto output_lrn = testLrnOp.predict_lrn(inputx1);
    auto *output_lrn_ptr = output_lrn->data<float>();

    DLOG << " LrnOp input: ";
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            for (int c = 0; c < 2; c++) {
                for (int d = 0; d < 2; d++) {
                    DLOGF("%f ", inputx1_ptr[i * 16 + j * 4 + c * 2 + d]);
                }
                DLOGF("\n");
            }
            DLOGF("\n");
        }
        DLOGF("\n");
    }
    DLOG << " LrnOp output: ";
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            for (int c = 0; c < 2; c++) {
                for (int d = 0; d < 2; d++) {
                    DLOGF("%f ", output_lrn_ptr[i * 16 + j * 4 + c * 2 + d]);
                }
                DLOGF("\n");
            }
            DLOGF("\n");
        }
        DLOGF("\n");
    }
    DLOG << inputx1_ptr[0] << " / ((1 + 0.00002 * ( " << inputx1_ptr[0]
         << "^2 + " << inputx1_ptr[4] << "^2 + " << inputx1_ptr[8]
         << "^2 ))^0.75) = ";
    DLOG << output_lrn_ptr[0];
    return 0;
}
