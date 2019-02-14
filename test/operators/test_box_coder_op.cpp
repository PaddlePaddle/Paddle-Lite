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
#include "operators/box_coder_op.h"

namespace paddle_mobile {
namespace framework {

template <typename Dtype>
class TestBoxCoderOp {
 public:
  explicit TestBoxCoderOp(const Program<Dtype> p) : program_(p) {
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
        if (op->Type() == "box_coder" &&
            op->Input("PriorBox")[0] == "concat_0.tmp_0") {
          DLOG << " mul attr size: " << op->GetAttrMap().size();
          DLOG << " inputs size: " << op->GetInputs().size();
          DLOG << " outputs size: " << op->GetOutputs().size();
          DLOG << " Input PriorBox is : " << op->Input("PriorBox")[0];
          DLOG << " Input PriorBoxVar is : " << op->Input("PriorBoxVar")[0];
          DLOG << " Input TargetBox is : " << op->Input("TargetBox")[0];
          DLOG << " OutputBox is : " << op->Output("OutputBox")[0];
          DLOG << " code_type : "
               << op->GetAttrMap().at("code_type").GetString();
          std::shared_ptr<operators::BoxCoderOp<Dtype, float>> boxcoder =
              std::make_shared<operators::BoxCoderOp<Dtype, float>>(
                  op->Type(), op->GetInputs(), op->GetOutputs(),
                  op->GetAttrMap(), program_.scope.get());
          ops_of_block_[*block_desc.get()].push_back(boxcoder);
        }
      }
    }
  }

  std::shared_ptr<Tensor> predict_boxcoder(const Tensor &t1, const Tensor &t2,
                                           const Tensor &t3) {
    // feed
    auto scope = program_.scope.get();
    Variable *prior_box = scope->Var("concat_0.tmp_0");
    auto tensor_x1 = prior_box->GetMutable<LoDTensor>();
    tensor_x1->ShareDataWith(t1);

    Variable *prior_box_var = scope->Var("concat_1.tmp_0");
    auto tensor_x2 = prior_box_var->GetMutable<LoDTensor>();
    tensor_x2->ShareDataWith(t2);

    Variable *target_box = scope->Var("concat_2.tmp_0");
    auto tensor_x3 = target_box->GetMutable<LoDTensor>();
    tensor_x3->ShareDataWith(t3);

    Variable *boxes_output = scope->Var("box_coder_0.tmp_0");
    auto *boxes_output_tensor = boxes_output->GetMutable<LoDTensor>();
    boxes_output_tensor->mutable_data<float>({1, 1917, 4});

    //  DLOG << typeid(output_tensor).name();
    //  DLOG << "output_tensor dims: " << output_tensor->dims();

    std::shared_ptr<Tensor> outbox_tensor = std::make_shared<LoDTensor>();
    outbox_tensor.reset(boxes_output_tensor);

    predict_boxcoder(t1, t2, t3, 0);

    return outbox_tensor;
  }

 private:
  const framework::Program<Dtype> program_;
  std::shared_ptr<ProgramDesc> to_predict_program_;
  std::map<framework::BlockDesc,
           std::vector<std::shared_ptr<OperatorBase<Dtype>>>>
      ops_of_block_;
  bool use_optimize_ = false;

  void predict_boxcoder(const Tensor &t1, const Tensor &t2, const Tensor &t3,
                        int block_id) {
    std::shared_ptr<BlockDesc> to_predict_block =
        to_predict_program_->Block(block_id);
    for (int j = 0; j < ops_of_block_[*to_predict_block.get()].size(); ++j) {
      auto op = ops_of_block_[*to_predict_block.get()][j];
      DLOG << "op -> run()";
      op->Run();
    }
  }
};

template class TestBoxCoderOp<CPU>;
}  // namespace framework
}  // namespace paddle_mobile

int main() {
  DLOG << "----------**********----------";
  DLOG << "begin to run BoxCoderOp Test";
  paddle_mobile::framework::Loader<paddle_mobile::CPU> loader;
  auto program = loader.Load(std::string(g_mobilenet_ssd));

  paddle_mobile::framework::Tensor priorbox;
  SetupTensor<float>(&priorbox, {1917, 4}, static_cast<float>(0),
                     static_cast<float>(1));
  auto *priorbox_ptr = priorbox.data<float>();

  paddle_mobile::framework::Tensor priorboxvar;
  SetupTensor<float>(&priorboxvar, {1917, 4}, static_cast<float>(0.1),
                     static_cast<float>(0.2));
  auto *priorboxvar_ptr = priorboxvar.data<float>();

  paddle_mobile::framework::Tensor targetbox;
  SetupTensor<float>(&targetbox, {1, 1917, 4}, static_cast<float>(0),
                     static_cast<float>(1));
  auto *targetbox_ptr = targetbox.data<float>();

  paddle_mobile::framework::TestBoxCoderOp<paddle_mobile::CPU> testBoxCoderOp(
      program);

  auto output_boxcoder =
      testBoxCoderOp.predict_boxcoder(priorbox, priorboxvar, targetbox);
  auto output_boxcoder_ptr = output_boxcoder->data<float>();

  for (int i = 0; i < output_boxcoder->numel(); i++) {
    DLOG << output_boxcoder_ptr[i];
  }
  DLOGF("\n");
  /// testing 25th bbox.
  DLOG << "PriorBox**************";
  DLOG << priorbox_ptr[100];
  DLOG << priorbox_ptr[101];
  DLOG << priorbox_ptr[102];
  DLOG << priorbox_ptr[103];
  DLOG << "PriorBoxVar**************";
  DLOG << priorboxvar_ptr[100];
  DLOG << priorboxvar_ptr[101];
  DLOG << priorboxvar_ptr[102];
  DLOG << priorboxvar_ptr[103];
  DLOG << "TargetBox***************";
  DLOG << targetbox_ptr[100];
  DLOG << targetbox_ptr[101];
  DLOG << targetbox_ptr[102];
  DLOG << targetbox_ptr[103];
  DLOG << "OutputBox**************";
  DLOG << output_boxcoder_ptr[100];
  DLOG << output_boxcoder_ptr[101];
  DLOG << output_boxcoder_ptr[102];
  DLOG << output_boxcoder_ptr[103];

  DLOG << "***********----------------------**************";
  auto priorbox_w = priorbox_ptr[102] - priorbox_ptr[100];
  auto priorbox_h = priorbox_ptr[103] - priorbox_ptr[101];
  auto priorbox_center_x = (priorbox_ptr[100] + priorbox_ptr[102]) / 2;
  auto priorbox_center_y = (priorbox_ptr[101] + priorbox_ptr[103]) / 2;
  DLOG << "prior box width : " << priorbox_w;
  DLOG << "prior box height : " << priorbox_h;
  DLOG << "prior box center x : " << priorbox_center_x;
  DLOG << "prior box center y : " << priorbox_center_y;
  auto target_box_center_x =
      priorboxvar_ptr[100] * targetbox_ptr[100] * priorbox_w +
      priorbox_center_x;
  DLOG << "target_box_center_x : " << target_box_center_x;
  auto target_box_center_y =
      priorboxvar_ptr[101] * targetbox_ptr[101] * priorbox_h +
      priorbox_center_y;
  DLOG << "target_box_center_y : " << target_box_center_y;
  auto target_box_width =
      std::exp(priorboxvar_ptr[102] * targetbox_ptr[102]) * priorbox_w;
  DLOG << "target_box_width : " << target_box_width;
  auto target_box_height =
      std::exp(priorboxvar_ptr[103] * targetbox_ptr[103]) * priorbox_h;
  DLOG << "target_box_height : " << target_box_height;
  DLOG << "pre x min : " << target_box_center_x - target_box_width / 2;
  DLOG << "pre y min : " << target_box_center_y - target_box_height / 2;
  DLOG << "pre x max : " << target_box_center_x + target_box_width / 2;
  DLOG << "pre y max : " << target_box_center_y + target_box_height / 2;
  return 0;
}
