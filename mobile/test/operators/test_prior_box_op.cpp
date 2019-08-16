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

#include "../test_include.h"
#include "operators/prior_box_op.h"

namespace paddle_mobile {
namespace framework {

template <typename Dtype>
class TestPriorBoxOp {
 public:
  explicit TestPriorBoxOp(const Program<Dtype> p) : program_(p) {
    if (use_optimize_) {
      to_predict_program_ = program_.optimizeProgram;
    } else {
      to_predict_program_ = program_.originProgram;
    }

    const std::vector<std::shared_ptr<BlockDesc>> blocks =
        to_predict_program_->Blocks();
    //  DLOG << " **block size " << blocks.size();
    for (auto block_desc : blocks) {
      std::vector<std::shared_ptr<OpDesc>> ops = block_desc->Ops();
      //    DLOG << " ops " << ops.size();
      for (auto op : ops) {
        if (op->Type() == "prior_box" &&
            op->Input("Input")[0] == "batch_norm_26.tmp_3") {
          DLOG << " mul attr size: " << op->GetAttrMap().size();
          DLOG << " inputs size: " << op->GetInputs().size();
          DLOG << " outputs size: " << op->GetOutputs().size();
          DLOG << " Input is : " << op->Input("Input")[0];
          DLOG << " Image is : " << op->Input("Image")[0];
          DLOG << " Output Boxes is : " << op->Output("Boxes")[0];
          DLOG << " Output Variances is : " << op->Output("Variances")[0];
          DLOG << " offset : " << op->GetAttrMap().at("offset").Get<float>();
          DLOG << " step_h : " << op->GetAttrMap().at("step_h").Get<float>();
          DLOG << " step_w : " << op->GetAttrMap().at("step_w").Get<float>();
          DLOG << " flip : " << op->GetAttrMap().at("flip").Get<bool>();
          DLOG << " clip : " << op->GetAttrMap().at("clip").Get<bool>();
          //                            DLOG << " variances : " <<
          //                            op->GetAttrMap().at("variances").Get<std::vector<float>>();
          //                            DLOG << " aspect_ratios : " <<
          //                            op->GetAttrMap().at("aspect_ratios").Get<std::vector<float>>();
          //                            DLOG << " min_sizes : " <<
          //                            op->GetAttrMap().at("min_sizes").Get<std::vector<float>>();
          //                            DLOG << " max_sizes : " <<
          //                            op->GetAttrMap().at("max_sizes").Get<std::vector<float>>();
          std::shared_ptr<operators::PriorBoxOp<Dtype, float>> priorbox =
              std::make_shared<operators::PriorBoxOp<Dtype, float>>(
                  op->Type(), op->GetInputs(), op->GetOutputs(),
                  op->GetAttrMap(), program_.scope.get());
          ops_of_block_[*block_desc.get()].push_back(priorbox);
        }
      }
    }
  }

  std::shared_ptr<Tensor> predict_priorbox(const Tensor &t1, const Tensor &t2) {
    // feed
    auto scope = program_.scope.get();
    Variable *x1_feed_value = scope->Var("image");
    auto tensor_x1 = x1_feed_value->GetMutable<LoDTensor>();
    tensor_x1->ShareDataWith(t1);

    Variable *x2_feed_value = scope->Var("batch_norm_26.tmp_3");
    auto tensor_x2 = x2_feed_value->GetMutable<LoDTensor>();
    tensor_x2->ShareDataWith(t2);

    Variable *boxes_output = scope->Var("prior_box_1.tmp_0");
    auto *boxes_output_tensor = boxes_output->GetMutable<LoDTensor>();
    boxes_output_tensor->mutable_data<float>({10, 10, 6, 4});

    Variable *variances_output = scope->Var("prior_box_1.tmp_1");
    auto *variances_output_tesnor = variances_output->GetMutable<LoDTensor>();
    variances_output_tesnor->mutable_data<float>({10, 10, 6, 4});
    //  DLOG << typeid(output_tensor).name();
    //  DLOG << "output_tensor dims: " << output_tensor->dims();

    std::shared_ptr<Tensor> outboxes_tensor = std::make_shared<LoDTensor>();
    outboxes_tensor.reset(boxes_output_tensor);

    std::shared_ptr<Tensor> outvars_tensor = std::make_shared<LoDTensor>();
    outvars_tensor.reset(variances_output_tesnor);
    predict_priorbox(t1, t2, 0);

    return outboxes_tensor;
    // return outvars_tensor;
  }

 private:
  const framework::Program<Dtype> program_;
  std::shared_ptr<ProgramDesc> to_predict_program_;
  std::map<framework::BlockDesc,
           std::vector<std::shared_ptr<OperatorBase<Dtype>>>>
      ops_of_block_;
  bool use_optimize_ = false;

  void predict_priorbox(const Tensor &t1, const Tensor &t2, int block_id) {
    std::shared_ptr<BlockDesc> to_predict_block =
        to_predict_program_->Block(block_id);
    for (int j = 0; j < ops_of_block_[*to_predict_block.get()].size(); ++j) {
      auto op = ops_of_block_[*to_predict_block.get()][j];
      DLOG << "op -> run()";
      op->Run();
    }
  }
};

template class TestPriorBoxOp<CPU>;
}  // namespace framework
}  // namespace paddle_mobile

int main() {
  DLOG << "----------**********----------";
  DLOG << "begin to run PriorBoxOp Test";
  paddle_mobile::framework::Loader<paddle_mobile::CPU> loader;
  auto program = loader.Load(std::string(g_mobilenet_ssd));

  /// input x (1,3,300,300)
  paddle_mobile::framework::Tensor input_image;
  SetupTensor<float>(&input_image, {1, 3, 300, 300}, static_cast<float>(0),
                     static_cast<float>(1));
  auto *input_image_ptr = input_image.data<float>();

  paddle_mobile::framework::Tensor inputx1;
  SetupTensor<float>(&inputx1, {1, 1024, 10, 10}, static_cast<float>(0),
                     static_cast<float>(1));
  auto *inputx1_ptr = inputx1.data<float>();

  paddle_mobile::framework::TestPriorBoxOp<paddle_mobile::CPU> testPriorBoxOp(
      program);

  auto output_priorbox = testPriorBoxOp.predict_priorbox(input_image, inputx1);
  auto *output_priorbox_ptr = output_priorbox->data<float>();

  for (int i = 0; i < output_priorbox->numel(); i++) {
    DLOG << output_priorbox_ptr[i];
  }
  return 0;
}
