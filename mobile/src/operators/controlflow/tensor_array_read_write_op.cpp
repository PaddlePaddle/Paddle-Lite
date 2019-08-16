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

#include "operators/controlflow/tensor_array_read_write_op.h"

namespace paddle_mobile {
namespace operators {

#ifdef WRITE_TO_ARRAY_OP
template <typename Dtype, typename T>
void WriteToArrayOp<Dtype, T>::InferShape() const {}
#endif  // WRITE_TO_ARRAY_OP

#ifdef READ_FROM_ARRAY_OP
template <typename Dtype, typename T>
void ReadFromArrayOp<Dtype, T>::InferShape() const {}
#endif  // READ_FROM_ARRAY_OP

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;

#ifdef PADDLE_MOBILE_CPU
#ifdef WRITE_TO_ARRAY_OP
REGISTER_OPERATOR_CPU(write_to_array, ops::WriteToArrayOp);
#endif  // WRITE_TO_ARRAY_OP

#ifdef READ_FROM_ARRAY_OP
REGISTER_OPERATOR_CPU(read_from_array, ops::ReadFromArrayOp);
#endif  // READ_FROM_ARRAY_OP
#endif
