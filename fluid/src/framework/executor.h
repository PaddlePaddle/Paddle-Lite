
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

#include <map>
#include <string>
#include <vector>
#include "program_desc.h"
#include "scope.h"
#include "variable.h"
#include "tensor.h"
#include "../operators/operator.h"
#include "framework.pb.h"

namespace paddle_mobile {
    namespace framework {
        extern void InitializeVariable(Variable *var, proto::VarType::Type var_type);

        struct ExecutorPrepareContext {
            ExecutorPrepareContext(const framework::ProgramDesc &prog, size_t block_id);

            ~ExecutorPrepareContext();

            const framework::ProgramDesc &prog_;
            size_t block_id_;
            std::vector<std::unique_ptr<OperatorBase> > ops_;
        };

        class Executor {
        public:
            explicit Executor();

            /* @Brief
             * Runtime evaluation of the given ProgramDesc under certain Scope
             *
             * @param
             *  ProgramDesc
             *  Scope
             */
            void Run(const ProgramDesc &prog, Scope *scope, int block_id,
                     bool create_local_scope = true, bool create_vars = true);


            static std::unique_ptr<ExecutorPrepareContext> Prepare(
                    const ProgramDesc &program, int block_id);

            static std::vector<std::shared_ptr<ExecutorPrepareContext> > Prepare(
                    const ProgramDesc &program, const std::vector<int> &block_ids);

            void CreateVariables(const ProgramDesc &pdesc, Scope *scope, int block_id);

            void RunPreparedContext(ExecutorPrepareContext *ctx, Scope *scope,
                                    bool create_local_scope = true,
                                    bool create_vars = true);


        };
    } // namesapce framework


}  // namespace paddle_mobile
