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
#include "operators/multiclass_nms_op.h"

namespace paddle_mobile {
namespace framework {

template <typename Dtype>
class TestMultiClassNMSOp {
 public:
  explicit TestMultiClassNMSOp(const Program<Dtype> p) : program_(p) {
    if (use_optimize_) {
      to_predict_program_ = program_.optimizeProgram;
    } else {
      to_predict_program_ = program_.originProgram;
    }

    const std::vector<std::shared_ptr<BlockDesc>> blocks =
        to_predict_program_->Blocks();
    for (auto block_desc : blocks) {
      std::vector<std::shared_ptr<OpDesc>> ops = block_desc->Ops();
      for (auto op : ops) {
        if (op->Type() == "multiclass_nms" &&
            op->Input("BBoxes")[0] == "box_coder_0.tmp_0") {
          DLOG << " attr size: " << op->GetAttrMap().size();
          DLOG << " inputs size: " << op->GetInputs().size();
          DLOG << " outputs size: " << op->GetOutputs().size();
          DLOG << " BBoxes is : " << op->Input("BBoxes")[0];
          DLOG << " Scores is : " << op->Input("Scores")[0];
          DLOG << " Out is : " << op->Output("Out")[0];
          DLOG << " keep_top_k : "
               << op->GetAttrMap().at("keep_top_k").Get<int>();
          DLOG << " background_label : "
               << op->GetAttrMap().at("background_label").Get<int>();
          DLOG << " nms_eta : " << op->GetAttrMap().at("nms_eta").Get<float>();
          DLOG << " nms_threshold : "
               << op->GetAttrMap().at("nms_threshold").Get<float>();
          DLOG << " nms_top_k : "
               << op->GetAttrMap().at("nms_top_k").Get<int>();
          DLOG << " score_threshold : "
               << op->GetAttrMap().at("score_threshold").Get<float>();
          std::shared_ptr<operators::MultiClassNMSOp<Dtype, float>> priorbox =
              std::make_shared<operators::MultiClassNMSOp<Dtype, float>>(
                  op->Type(), op->GetInputs(), op->GetOutputs(),
                  op->GetAttrMap(), program_.scope.get());
          ops_of_block_[*block_desc.get()].push_back(priorbox);
        }
      }
    }
  }

  std::shared_ptr<Tensor> predict(const Tensor &t1, const Tensor &t2) {
    // feed
    auto scope = program_.scope.get();
    Variable *x1_feed_value = scope->Var("box_coder_0.tmp_0");
    auto tensor_x1 = x1_feed_value->GetMutable<LoDTensor>();
    tensor_x1->ShareDataWith(t1);

    Variable *x2_feed_value = scope->Var("transpose_12.tmp_0");
    auto tensor_x2 = x2_feed_value->GetMutable<LoDTensor>();
    tensor_x2->ShareDataWith(t2);

    Variable *output = scope->Var("detection_output_0.tmp_0");
    auto *output_tensor = output->GetMutable<LoDTensor>();
    output_tensor->mutable_data<float>({1917, 6});

    std::shared_ptr<Tensor> out_tensor = std::make_shared<LoDTensor>();
    out_tensor.reset(output_tensor);

    predict(t1, t2, 0);

    return out_tensor;
  }

 private:
  const framework::Program<Dtype> program_;
  std::shared_ptr<ProgramDesc> to_predict_program_;
  std::map<framework::BlockDesc,
           std::vector<std::shared_ptr<OperatorBase<Dtype>>>>
      ops_of_block_;
  bool use_optimize_ = false;

  void predict(const Tensor &t1, const Tensor &t2, int block_id) {
    std::shared_ptr<BlockDesc> to_predict_block =
        to_predict_program_->Block(block_id);
    for (int j = 0; j < ops_of_block_[*to_predict_block.get()].size(); ++j) {
      auto op = ops_of_block_[*to_predict_block.get()][j];
      DLOG << "op -> run()";
      op->Run();
    }
  }
};

template class TestMultiClassNMSOp<CPU>;
}  // namespace framework
}  // namespace paddle_mobile

int main() {
  DLOG << "----------**********----------";
  DLOG << "begin to run MulticlassNMS Test";
  paddle_mobile::framework::Loader<paddle_mobile::CPU> loader;
  auto program = loader.Load(std::string(g_mobilenet_ssd));
  paddle_mobile::framework::Tensor inputx1;
  SetupTensor<float>(&inputx1, {1, 2, 4}, static_cast<float>(0),
                     static_cast<float>(1));
  auto *inputx1_ptr = inputx1.data<float>();
  const float x1[] = {0, 0, 100, 100, 50, 50, 150, 150};
  for (int i = 0; i < 8; ++i) {
    *(inputx1_ptr + i) = x1[i];
  }

  paddle_mobile::framework::Tensor inputx2;
  SetupTensor<float>(&inputx2, {1, 2, 2}, static_cast<float>(0),
                     static_cast<float>(1));
  auto *inputx2_ptr = inputx2.data<float>();
  const float x2[] = {0.4, 0.3, 0.6, 0.7};
  for (int i = 0; i < 4; ++i) {
    *(inputx2_ptr + i) = x2[i];
  }

  paddle_mobile::framework::TestMultiClassNMSOp<paddle_mobile::CPU>
      testMultiClassNMSOp(program);

  auto output = testMultiClassNMSOp.predict(inputx1, inputx2);
  auto *output_ptr = output->data<float>();

  for (int i = 0; i < output->numel(); ++i) {
    DLOG << output_ptr[i];
  }

  // test multi point
  paddle_mobile::framework::Tensor inputx3;
  SetupTensor<float>(&inputx3, {1, 2, 8}, static_cast<float>(0),
                     static_cast<float>(1));
  auto *inputx3_ptr = inputx3.data<float>();
  const float x3[] = {0,  0,  100, 0,  100, 100, 0,  100,
                      50, 50, 150, 50, 150, 150, 50, 150};
  for (int i = 0; i < 16; ++i) {
    *(inputx3_ptr + i) = x3[i];
  }

  auto output2 = testMultiClassNMSOp.predict(inputx3, inputx2);
  auto *output_ptr2 = output2->data<float>();

  for (int i = 0; i < output2->numel(); ++i) {
    DLOG << output_ptr2[i];
  }
  return 0;
}
