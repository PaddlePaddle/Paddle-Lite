
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
#include "operators/conv_op.h"

using std::cout;
namespace paddle_mobile {

namespace framework {

namespace {
// block id starts from 0. This id is used to represent the codeblock
// wrapping the first block 0.
    int kProgramId = -1;
}  // namespace
template <typename Dtype>
ExecutorPrepareContext<Dtype>::ExecutorPrepareContext(
        const framework::ProgramDesc &prog, size_t block_id)
        : prog_(prog), block_id_(block_id) {}

template <typename Dtype>
ExecutorPrepareContext<Dtype>::~ExecutorPrepareContext() {
    cout << "destroy ExecutorPrepareContext";
}

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
//          operators::ConvOp<Dtype> conv(op->Type(), op->GetInputs(), op->GetOutputs(), op->GetAttrMap());

        }
      }
    }
  }
}


void InitializeVariable(Variable *var, proto::VarType::Type var_type) {

}

static void CheckTensorNANOrInf(const std::string &name,
                                const Tensor &tensor) {

}

template <typename Dtype>
void Executor<Dtype>::CreateVariables(const ProgramDesc &pdesc, Scope *scope,
                               int block_id) {

}

template <typename Dtype>
void Executor<Dtype>::Run(const ProgramDesc &pdesc, Scope *scope, int block_id,
                   bool create_local_scope, bool create_vars) {

}

template <typename Dtype>
std::unique_ptr<ExecutorPrepareContext<Dtype> > Executor<Dtype>::Prepare(
        const ProgramDesc &program, int block_id) {
}

template <typename Dtype>
std::vector<std::shared_ptr<ExecutorPrepareContext<Dtype> > > Executor<Dtype>::Prepare(
        const ProgramDesc &program, const std::vector<int> &block_ids) {

}

template <typename Dtype>
void Executor<Dtype>::RunPreparedContext(ExecutorPrepareContext<Dtype> *ctx, Scope *scope,
                                  bool create_local_scope, bool create_vars) {

}

}

}  // namespace paddle



