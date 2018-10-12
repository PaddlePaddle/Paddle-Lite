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


#include "operators/kernel/feed_kernel.h"

namespace paddle_mobile {
    namespace operators {

        template <>
        bool FeedKernel<GPU_CL, float>::Init(FeedParam<GPU_CL> *param) {
            DLOG<<"Init feed";
         this->cl_helper_.AddKernel("feed", "feed_kernel.cl");
            return true;
        }

        template <>
        void FeedKernel<GPU_CL, float>::Compute(const FeedParam<GPU_CL> &param) {

         DLOG<<"feed_kernel";
         auto kernel = this->cl_helper_.KernelAt(0);
         cl_int status;
         auto output = param.Out();
         auto input = param.InputX();
         const float *input_data = input->data<float>();
         cl_mem cl_image = output->GetCLImage();
         int height = output->dims()[2];
         int width = output->dims()[3];
         status = clSetKernelArg(kernel,0, sizeof(cl_mem),&input_data);
         status = clSetKernelArg(kernel,0, sizeof(cl_mem),&cl_image);
         status = clSetKernelArg(kernel,0, sizeof(cl_mem),&width);
         status = clSetKernelArg(kernel,0, sizeof(cl_mem),&height);

         size_t global_work_size[2] = {height,width};
         clEnqueueNDRangeKernel(this->cl_helper_.CLCommandQueue(), kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
        }

        template class FeedKernel<GPU_CL, float>;

    }  // namespace operators
}  // namespace paddle_mobile

