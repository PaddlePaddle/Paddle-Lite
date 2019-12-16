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
#include "bmcompiler_if.h"
#include "bmcompiler_if_lite.h"
#include "bmcompiler_defs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace bm {
namespace bridges {

node_map_type ElementwiseConverter(const std::shared_ptr<lite::OpLite> elementwise_op,
                            graph_ctx_type* graph_ctx,
                            const node_map_type& input_nodes) {
    // output converted nodes
    node_map_type output_nodes;
    auto scope = elementwise_op->scope();
    auto op_info = elementwise_op->op_info();
    auto op_type = op_info->Type();
    
    // input
    const int input_num = 2;
    int **shape = new int *[input_num];
    int *dim = new int[input_num];
    const char **name = new const char *[input_num];
    
    auto x_var_name = op_info->Input("X").front();
    auto x = scope->FindVar(x_var_name)->GetMutable<lite::Tensor>();
    auto x_dims = x->dims();
    name[0] = static_cast<const char*>(x_var_name.c_str());
    dim[0] = x_dims.size();
    const long int* x_shape_data = const_cast<const long int*>(&x_dims.data()[0]);
    int i_x_shape_data[x_dims.size()];
    for (size_t i = 0; i < x_dims.size(); i++) {
        i_x_shape_data[i] = static_cast<int>(x_shape_data[i]);
    }
    shape[0] = i_x_shape_data;
    
    auto y_var_name = op_info->Input("Y").front();
    auto y = scope->FindVar(y_var_name)->GetMutable<lite::Tensor>();
    auto y_dims = y->dims();
    name[1] = static_cast<const char*>(y_var_name.c_str());
    dim[1] = y_dims.size();
    const long int* y_shape_data = const_cast<const long int*>(&y_dims.data()[0]);
    int i_y_shape_data[y_dims.size()];
    for (size_t i = 0; i < y_dims.size(); i++) {
        i_y_shape_data[i] = static_cast<int>(y_shape_data[i]);
    }
    shape[1] = i_y_shape_data;
    bool y_is_const = input_nodes.find(y_var_name) == input_nodes.end();
   
    // output
    auto output_var_name = op_info->Output("Out").front();
    auto output = scope->FindVar(output_var_name)->GetMutable<lite::Tensor>();
    auto output_dims = output->dims();
    const long int* output_shape_data = const_cast<const long int*>(&output_dims.data()[0]);
    int i_output_shape_data[output_dims.size()];
    for (size_t i = 0; i < output_dims.size(); i++) {
        i_output_shape_data[i] = static_cast<int>(output_shape_data[i]);
    }
    
    if (y_is_const) {
        CHECK(op_type == "elementwise_add");
    }
    
    int op_code{-1};
    float coeff[2] = {1.f, 1.f};

    if (op_type == "elementwise_mul") {
        op_code = 0;
    } else if (op_type == "elementwise_add") {
        op_code = 1;
    } else if(op_type == "elementwise_sub") {
        op_code = 1;
        coeff[1] = -1.f;
    } else {
        LOG(FATAL) << "UNSUPPORTED ELTWISE OPERATION: " << op_type;
    }
    
    if (!y_is_const) {
        add_eltwise_layer(graph_ctx->bm_compiler_handle,
                      input_num,
                      shape,
                      dim,
                      name,
                      const_cast<const int*>(i_output_shape_data),
                      output_dims.size(),
                      static_cast<const char*>(output_var_name.c_str()),
                      op_code,
                      coeff);
    } else {
        const float* y_data = const_cast<const float*>(y->mutable_data<float>());
        const float* x_data = const_cast<const float*>(x->mutable_data<float>());
        bm_add_const_tensor(graph_ctx->bm_compiler_handle,
                            name[1],
                            shape[0],
                            dim[0],
                            static_cast<bm_data_type_t>(DTYPE_FP32),
                            static_cast<const void*>(y_data));

        add_binary_layer_v2(graph_ctx->bm_compiler_handle,
                          name[0],
                          shape[0],
                          dim[0],
                          0,
                          static_cast<const float*>(x_data),
                          name[1],
                          shape[0],
                          dim[0],
                          0,
                          static_cast<const float*>(y_data),
                          static_cast<const char*>(output_var_name.c_str()),
                          0);
    }

    delete [] shape;
    delete [] name;
    delete [] dim;
    
    output_nodes[output_var_name] = output_var_name;
    return output_nodes;
}

}  // namespace bridges
}  // namespace bm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_BM_BRIDGE(elementwise_add, paddle::lite::kernels::bm::bridges::ElementwiseConverter);
