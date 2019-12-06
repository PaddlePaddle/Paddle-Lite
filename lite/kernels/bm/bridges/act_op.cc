// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/kernels/bm/bridges/registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace bm {
namespace bridges {

node_map_type ActConverter(const std::shared_ptr<lite::OpLite> op,
                            const node_map_type& input_nodes) {
  // output converted nodes
  node_map_type output_nodes;
  return output_nodes;
}

}  // namespace bridges
}  // namespace bm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_BM_BRIDGE(relu, paddle::lite::kernels::bm::bridges::ActConverter);
