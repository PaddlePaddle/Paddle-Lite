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

#pragma once
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
    //  DLOG << " **block size " << blocks.size();
    for (auto block_desc : blocks) {
      std::vector<std::shared_ptr<OpDesc>> ops = block_desc->Ops();
      //    DLOG << " ops " << ops.size();
      for (auto op : ops) {
        if (op->Type() == "multiclass_nms" &&
            op->Input("BBoxes")[0] == "box_coder_0.tmp_0") {
          DLOG << " mul attr size: " << op->GetAttrMap().size();
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
          //                            DLOG << " variances : " <<
          //                            op->GetAttrMap().at("variances").Get<std::vector<float>>();
          //                            DLOG << " aspect_ratios : " <<
          //                            op->GetAttrMap().at("aspect_ratios").Get<std::vector<float>>();
          //                            DLOG << " min_sizes : " <<
          //                            op->GetAttrMap().at("min_sizes").Get<std::vector<float>>();
          //                            DLOG << " max_sizes : " <<
          //                            op->GetAttrMap().at("max_sizes").Get<std::vector<float>>();
          std::shared_ptr<operators::MultiClassNMSOp<Dtype, float>> priorbox =
              std::make_shared<operators::MultiClassNMSOp<Dtype, float>>(
                  op->Type(), op->GetInputs(), op->GetOutputs(),
                  op->GetAttrMap(), program_.scope);
          ops_of_block_[*block_desc.get()].push_back(priorbox);
        }
      }
    }
  }

  std::shared_ptr<Tensor> predict(const Tensor &t1, const Tensor &t2) {
    // feed
    auto scope = program_.scope;
    Variable *x1_feed_value = scope->Var("box_coder_0.tmp_0");
    auto tensor_x1 = x1_feed_value->GetMutable<LoDTensor>();
    tensor_x1->ShareDataWith(t1);

    Variable *x2_feed_value = scope->Var("transpose_12.tmp_0");
    auto tensor_x2 = x2_feed_value->GetMutable<LoDTensor>();
    tensor_x2->ShareDataWith(t2);

    Variable *output = scope->Var("detection_output_0.tmp_0");
    auto *output_tensor = output->GetMutable<LoDTensor>();
    output_tensor->mutable_data<float>({1917, 6});

    //  DLOG << typeid(output_tensor).name();
    //  DLOG << "output_tensor dims: " << output_tensor->dims();

    std::shared_ptr<Tensor> out_tensor = std::make_shared<LoDTensor>();
    out_tensor.reset(output_tensor);

    predict(t1, t2, 0);

    return out_tensor;
    // return outvars_tensor;
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
  auto program = loader.Load(std::string("../../test/models/mobilenet+ssd"));

  /// input x (1,3,300,300)
  paddle_mobile::framework::Tensor inputx1;
  SetupTensor<float>(&inputx1, {10, 1917, 4}, static_cast<float>(0),
                     static_cast<float>(1));
  auto *inputx1_ptr = inputx1.data<float>();

  paddle_mobile::framework::Tensor inputx2;
  SetupTensor<float>(&inputx2, {10, 21, 1917}, static_cast<float>(0),
                     static_cast<float>(1));
  auto *inputx2_ptr = inputx2.data<float>();

  paddle_mobile::framework::TestMultiClassNMSOp<paddle_mobile::CPU>
      testMultiClassNMSOp(program);

  auto output = testMultiClassNMSOp.predict(inputx1, inputx2);
  auto *output_ptr = output->data<float>();

  for (int i = 0; i < output->numel(); i++) {
    DLOG << output_ptr[i];
  }
  return 0;
}
