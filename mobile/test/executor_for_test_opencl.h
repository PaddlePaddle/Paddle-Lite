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
#ifdef PADDLE_MOBILE_CL

#include <string>
#include <vector>
#include <memory>

#include "common/log.h"
#include "framework/cl/cl_helper.h"
#include "framework/cl/cl_tensor.h"
#include "framework/executor.h"
#include "framework/op_registry.h"
#include "operators/feed_op.h"
#include "operators/fetch_op.h"
#include "./test_helper.h"

using paddle_mobile::framework::BlockDesc;
using paddle_mobile::framework::DDim;
using paddle_mobile::framework::Executor;
using paddle_mobile::framework::LoDTensor;
using paddle_mobile::framework::OpDesc;
using paddle_mobile::framework::Program;
using paddle_mobile::framework::Tensor;
using paddle_mobile::framework::Variable;
using paddle_mobile::framework::OperatorBase;
using paddle_mobile::framework::AttributeMap;
using std::string;
using std::vector;
namespace paddle_mobile {
template <typename OpType>
class OpenClOpTester {
 public:
  OpenClOpTester() {
    framework::CLEngine::Instance()->setClPath("/data/local/tmp/bin");
    scope_ = std::make_shared<paddle_mobile::framework::Scope>();
    feed_clhelper_ = framework::CLHelper(scope_->GetCLScpoe());
    fetch_clhelper_ = framework::CLHelper(scope_->GetCLScpoe());
    this->feed_clhelper_.AddKernel("feed", "feed_kernel.cl");
    this->fetch_clhelper_.AddKernel("fetch", "fetch_kernel.cl");

    feed_var = scope_.get()->Var("feed");
    fetch_var = scope_.get()->Var("fetch");
    op_in_var = scope_.get()->Var("op_in");
    op_out_var = scope_.get()->Var("op_out");
  }

  void Predict(string op_type, DDim feed_dims, DDim fetch_dims,
               VariableNameMap inputs_feed, VariableNameMap outputs_feed,
               AttributeMap attrs_feed) {
    framework::CLImage *const op_in_cl_image =
        op_in_var->template GetMutable<framework::CLImage>();
    op_in_cl_image->Resize(feed_dims);
    op_in_cl_image->InitEmptyImage(feed_clhelper_.CLContext(),
                                   feed_clhelper_.CLCommandQueue(), feed_dims);
    framework::CLImage *const op_out_cl_image =
        op_out_var->template GetMutable<framework::CLImage>();
    op_out_cl_image->Resize(fetch_dims);
    framework::CLScope *const clScpoe = scope_->GetCLScpoe();
    op_out_cl_image->InitEmptyImage(clScpoe->Context(), clScpoe->CommandQueue(),
                                    fetch_dims);

    Feed(feed_dims);
    auto *op = new OpType(op_type, inputs_feed, outputs_feed, attrs_feed,
                          scope_.get());
    op->InferShape();
    op->Init();
    op->Run();
    Fetch(fetch_dims);
  }
  void Feed(DDim feed_dims) {
    auto *feed_var = scope_->Var("feed");
    auto *_var = scope_->Var("op_in");
    auto *const input = feed_var->template GetMutable<framework::LoDTensor>();
    DLOG << "feed_dims: " << feed_dims;
    SetupTensor<float>(input, feed_dims, -100.0, 100.0);
    framework::CLImage *const op_in_cl_image =
        op_in_var->template GetMutable<framework::CLImage>();
    DLOG << "FeedKernel run ";
    DLOG << "params.input " << *input;
    DLOG << "params.op_in_cl_image " << *op_in_cl_image;
    auto kernel = this->feed_clhelper_.KernelAt(0);
    DLOG << "kernel get success ";

    auto default_work_size =
        this->feed_clhelper_.DefaultWorkSize(*(op_in_cl_image));

    DLOG << "op_in_cl_image: " << *op_in_cl_image;
    DLOG << "default_work_size: " << default_work_size;
    cl_int status;
    int numel = input->numel();
    cl_mem output_image = op_in_cl_image->GetCLImage();
    const int out_C = op_in_cl_image->dims()[1];
    const int out_H = op_in_cl_image->dims()[2];
    const int out_W = op_in_cl_image->dims()[3];
    const int Stride2 = out_C * out_H * out_W;
    const int Stride1 = out_H * out_W;
    const int Stride0 = out_W;
    framework::CLTensor input_cl_tensor(this->feed_clhelper_.CLContext(),
                                        this->feed_clhelper_.CLCommandQueue());
    input_cl_tensor.Resize(input->dims());
    cl_mem inputBuffer;

    inputBuffer =
        input_cl_tensor.mutable_with_data<float>(input->data<float>());

    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_image);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 2, sizeof(cl_int), &out_H);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 3, sizeof(cl_int), &out_W);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 4, sizeof(cl_int), &out_C);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 5, sizeof(cl_int), &Stride0);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 6, sizeof(cl_int), &Stride1);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 7, sizeof(cl_int), &Stride2);
    CL_CHECK_ERRORS(status);

    status = clEnqueueNDRangeKernel(
        this->feed_clhelper_.CLCommandQueue(), kernel, default_work_size.size(),
        NULL, default_work_size.data(), NULL, 0, NULL, NULL);

    CL_CHECK_ERRORS(status);

    DLOG << "*op_in_cl_image: " << *op_in_cl_image;
  }

  void Fetch(DDim fetch_dims) {
    DLOG << "------------------  Fetch op ---------------------";

    DLOG << "------------------  Fetch op end ---------------------";
  }

 private:
  std::shared_ptr<paddle_mobile::framework::Scope> scope_;
  framework::CLHelper feed_clhelper_;
  framework::CLHelper fetch_clhelper_;

  Variable *feed_var;
  Variable *fetch_var;
  Variable *op_in_var;
  Variable *op_out_var;
};
}  // namespace paddle_mobile
#endif
