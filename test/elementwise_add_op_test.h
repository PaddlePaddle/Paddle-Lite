
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
#include "operators/elementwise_add_op.h"
#include "test_include.h"

namespace paddle_mobile {
    namespace framework {

        template <typename Dtype> class TestElementwiseAddOp {
          public:
            TestElementwiseAddOp(const Program<Dtype> p) : program_(p) {
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
                    std::vector<std::shared_ptr<OpDesc>> ops =
                        block_desc->Ops();
                    //    DLOG << " ops " << ops.size();
                    for (int j = 0; j < ops.size(); ++j) {
                        std::shared_ptr<OpDesc> op = ops[j];
                        //                        if (op->Type() ==
                        //                        "elementwise_add") {
                        //                            if
                        //                            (op->GetAttrMap().at("axis").Get<int>()
                        //                            != -1) {
                        //                                DLOG << "attr: axis =
                        //                                "
                        //                                     <<
                        //                                     op->GetAttrMap().at("axis").Get<int>();
                        //                            }
                        //                        }
                        //                        DLOG << "op:" << op->Type();
                        if (op->Type() == "elementwise_add" &&
                            op->Input("X")[0] == "batch_norm_2.tmp_2") {
                            DLOG << " elementwise_add attr size: "
                                 << op->GetAttrMap().size();
                            DLOG << " inputs size: " << op->GetInputs().size();
                            DLOG << " outputs size: "
                                 << op->GetOutputs().size();
                            DLOG << " Input X is : " << op->Input("X")[0];
                            DLOG << " Input Y is : " << op->Input("Y")[0];
                            DLOG << " Output Out is : " << op->Output("Out")[0];
                            Attribute axis_attr = op->GetAttrMap().at("axis");
                            int axis = axis_attr.Get<int>();
                            DLOG << " Attr axis is : " << axis;

                            std::shared_ptr<
                                operators::ElementwiseAddOp<Dtype, float>>
                                add = std::make_shared<
                                    operators::ElementwiseAddOp<Dtype, float>>(
                                    op->Type(), op->GetInputs(),
                                    op->GetOutputs(), op->GetAttrMap(),
                                    program_.scope);
                            ops_of_block_[*block_desc.get()].push_back(add);
                        }
                    }
                }
            }

            std::shared_ptr<Tensor> predict_add(Tensor &t1, Tensor &t2) {
                // feed
                auto scope = program_.scope;
                Variable *x_feed_value = scope->Var("batch_norm_2.tmp_2");
                auto tensor_x = x_feed_value->GetMutable<Tensor>();
                tensor_x->ShareDataWith(t1);

                Variable *y_feed_value = scope->Var("batch_norm_0.tmp_3");
                auto tensor_y = y_feed_value->GetMutable<Tensor>();
                tensor_y->ShareDataWith(t2);

                Variable *con_output = scope->Var("elementwise_add_0.tmp_0");
                Tensor *output_tensor = con_output->GetMutable<Tensor>();
                output_tensor->mutable_data<float>({1, 3, 224, 224});
                //  DLOG << typeid(output_tensor).name();
                //  DLOG << "output_tensor dims: " << output_tensor->dims();

                std::shared_ptr<Tensor> out_tensor =
                    std::make_shared<LoDTensor>();
                out_tensor.reset(output_tensor);

                predict_add(t1, t2, 0);
                return out_tensor;
            }

          private:
            const framework::Program<Dtype> program_;
            std::shared_ptr<ProgramDesc> to_predict_program_;
            std::map<framework::BlockDesc,
                     std::vector<std::shared_ptr<OperatorBase<Dtype>>>>
                ops_of_block_;
            bool use_optimize_ = false;

            void predict_add(const Tensor &t1, const Tensor &t2, int block_id) {
                std::shared_ptr<BlockDesc> to_predict_block =
                    to_predict_program_->Block(block_id);
                for (int j = 0;
                     j < ops_of_block_[*to_predict_block.get()].size(); ++j) {
                    auto op = ops_of_block_[*to_predict_block.get()][j];
                    DLOG << "op -> run()";
                    op->Run();
                }
            }
        };

        template class TestElementwiseAddOp<CPU>;
    } // namespace framework

    namespace test {
        void testElementwiseAdd() {
            DLOG << "----------**********----------";
            DLOG << "begin to run ElementAddOp Test";
            paddle_mobile::Loader<paddle_mobile::CPU> loader;
            auto program = loader.Load(
                std::string("../../test/models/"
                            "image_classification_resnet.inference.model"));

            /// input x (1,3,224,224)
            paddle_mobile::framework::Tensor inputx;
            SetupTensor<float>(&inputx, {1, 3, 224, 224}, static_cast<float>(0),
                               static_cast<float>(1));
            float *inputx_ptr = inputx.data<float>();
            /// input y (224,)
            paddle_mobile::framework::Tensor inputy;
            SetupTensor<float>(&inputy, {224}, static_cast<float>(0),
                               static_cast<float>(1));
            float *inputy_ptr = inputy.data<float>();

            paddle_mobile::framework::TestElementwiseAddOp<paddle_mobile::CPU>
                testElementwiseAddOp(program);

            auto output_add = testElementwiseAddOp.predict_add(inputx, inputy);
            float *output_add_ptr = output_add->data<float>();
            //            for (int j = 0; j < output_add->numel(); ++j) {
            //                DLOG << "value of output: " << output_add_ptr[j];
            //            }

            /// output (1,3,224,224)
            DLOG << "output memory size : " << output_add->memory_size();
            DLOG << "output numel : " << output_add->numel();

            DLOG << inputx_ptr[226] << " + " << inputy_ptr[2] << " = "
                 << output_add_ptr[226];
        }
    } // namespace test
} // namespace paddle_mobile
