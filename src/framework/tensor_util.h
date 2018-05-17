/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include "framework.pb.h"
#include "memory/t_malloc.h"
#include "platform/data_type.h"
#include "tensor.h"
#include <vector>

namespace paddle_mobile {
    namespace framework {

        void TensorCopy(const Tensor &src, Tensor *dst);
        void TensorCopySync(const Tensor &src, Tensor *dst);

        template <typename T>
        void TensorFromVector(const std::vector<T> &src, Tensor *dst);

        template <typename T>
        void TesnorToVector(const Tensor &src, std::vector<T> *dst);

        bool TensorContainsNAN(const framework::Tensor &tensor);
        bool TensorContainsInf(const framework::Tensor &tensor);

        void TensorToStream(std::ostream &os, const Tensor &tensor);
        void TensorFromStream(std::istream &is, Tensor *tensor);

        //
        // The implementation of template functions.
        //

        template <typename T>
        void TensorFromVector(const std::vector<T> &src, Tensor *dst) {
            auto src_ptr = static_cast<const void *>(src.data());
            dst->Resize({static_cast<int64_t>(src.size())});
            auto dst_ptr = static_cast<void *>(dst->mutable_data<T>());
            auto size = src.size() * sizeof(T);

            memory::Copy(dst_ptr, src_ptr, size);
        }

        template <typename T>
        void TensorToVector(const Tensor &src, std::vector<T> *dst) {
            auto src_ptr = static_cast<const void *>(src.data<T>());
            auto size = src.numel() * sizeof(T);

            dst->resize(src.numel());
            auto dst_ptr = static_cast<void *>(dst->data());

            memory::Copy(dst_ptr, src_ptr, size);
        }

    } // namespace framework
} // namespace paddle_mobile
