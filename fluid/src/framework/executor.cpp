
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
using std::cout;
namespace paddle_mobile {
    namespace framework {

        namespace {
// block id starts from 0. This id is used to represent the codeblock
// wrapping the first block 0.
            int kProgramId = -1;
        }  // namespace

        ExecutorPrepareContext::ExecutorPrepareContext(
                const framework::ProgramDesc &prog, size_t block_id)
                : prog_(prog), block_id_(block_id) {}

        ExecutorPrepareContext::~ExecutorPrepareContext() {
            cout << "destroy ExecutorPrepareContext";
        }

        Executor::Executor() {}

        void InitializeVariable(Variable *var, proto::VarType::Type var_type) {

        }

        static void CheckTensorNANOrInf(const std::string &name,
                                        const Tensor &tensor) {

        }

        void Executor::CreateVariables(const ProgramDesc &pdesc, Scope *scope,
                                       int block_id) {

        }

        void Executor::Run(const ProgramDesc &pdesc, Scope *scope, int block_id,
                           bool create_local_scope, bool create_vars) {

        }

        std::unique_ptr<ExecutorPrepareContext> Executor::Prepare(
                const ProgramDesc &program, int block_id) {
        }

        std::vector<std::shared_ptr<ExecutorPrepareContext> > Executor::Prepare(
                const ProgramDesc &program, const std::vector<int> &block_ids) {

        }

        void Executor::RunPreparedContext(ExecutorPrepareContext *ctx, Scope *scope,
                                          bool create_local_scope, bool create_vars) {

        }
    }

}  // namespace paddle



