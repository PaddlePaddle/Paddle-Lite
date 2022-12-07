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

#include <gtest/gtest.h>
#include <cmath>

#include <gtest/gtest.h>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/test/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

void matrix_mul(int m_,
                int k_,
                int n_,
                float alpha,
                const float* x,
                const float* y,
                float* out) {
    for (int m = 0; m < m_; ++m) {
        for (int n = 0; n < n_; ++n) {
            out[m * n_ + n] = 0;
            for (int k = 0; k < k_; ++k) {
                out[m * n_ + n] += x[m * k_ + k] * y[k * n_ + n] * alpha;
            }
        }
    }
}

void matmul(float *x, float *y, float *out, std::vector<int> x_shape, std::vector<int> y_shape, float alpha, bool do_sfotmax=false) {
    xdnn::Context ctx_cpu(xdnn::kCPU);
    int batch = x_shape[0];
    int head_num = x_shape[1];
    int m = x_shape[2];
    int n = y_shape[3];
    int k = x_shape[3];

    for(size_t i = 0; i < batch; i++) {
        for (size_t j = 0; j < head_num; j++) {
           int x_shift = (i * (head_num * m * k)) + (j * (m * k));
           int y_shift = (i * (head_num * k * n)) + (j * (k * n));
           int output_shift = (i * (head_num * m * n)) + (j * (m * n)); 
           //matrix_mul(m, k, n, alpha, x + x_shift, y + y_shift, out + output_shift);
           
           if (do_sfotmax) {
                std::vector<float> tmp_out(m * n);
                matrix_mul(m, k, n, alpha, x + x_shift, y + y_shift, &tmp_out[0]);
                xdnn::softmax(&ctx_cpu, &tmp_out[0], out + output_shift, {m, n}, 1);
           } else {
                matrix_mul(m, k, n, alpha, x + x_shift, y + y_shift, out + output_shift);
           }
        }
    }
}

class QkvAttentionComputeTester : public arena::TestCase {
protected:
    // common attributes for this op.
    std::string input_q_ = "q";
    std::string input_k_ = "k";
    std::string input_v_ = "v";
    std::string output_ = "out";
    DDim dims_{{1, 2, 4, 3}};
    float scale_scale_ = 0.125;
    float scale_bias_ = 0;
    std::vector<int> transpose_axis_ = {0, 1, 3, 2};

public:
    QkvAttentionComputeTester(const Place& place,
                                const std::string& alias,
                                DDim dims)
      : TestCase(place, alias) {
        dims_ = dims;
        scale_scale_ = 1.0 / std::sqrt((float)dims_[3]);
    }

    void RunBaseline(Scope* scope) override {
        auto* out = scope->NewTensor(output_);

        CHECK(out);
        out->Resize(dims_);
        auto* out_data = out->mutable_data<float>();

        auto* q = scope->FindMutableTensor(input_q_);
        const auto* q_data = q->data<float>();
        auto* k = scope->FindMutableTensor(input_k_);
        const auto* k_data = k->data<float>();
        auto* v = scope->FindMutableTensor(input_v_);
        const auto* v_data = v->data<float>();

        std::vector<int> shape = {(int)dims_[0], (int)dims_[1], (int)dims_[2], (int)dims_[3]}; //[batch, head_num, seq_len, head_dim]
        xdnn::Context ctx_cpu(xdnn::kCPU);
        std::vector<float> transpose_k(shape[0] * shape[1] * shape[2] * shape[3]);
        int r = xdnn::transpose<float>(&ctx_cpu,
            k_data,
            (float*)&transpose_k[0],
            shape,
            transpose_axis_);
        CHECK_EQ(r, 0);

        std::vector<float> qk(shape[0] * shape[1] * shape[2] * shape[2]);
        matmul(const_cast<float*>(q_data), (float*)&transpose_k[0], (float*)&qk[0], shape, {shape[0], shape[1], shape[3], shape[2]}, scale_scale_, true);
        matmul((float*)&qk[0], const_cast<float*>(v_data), out_data, {shape[0], shape[1], shape[2], shape[2]}, shape, 1);
    }

    void PrepareOpDesc(cpp::OpDesc* op_desc) {

        op_desc->SetType("__xpu__qkv_attention");
        op_desc->SetInput("input_q", {input_q_});
        op_desc->SetInput("input_k", {input_k_});
        op_desc->SetInput("input_v", {input_v_});
        op_desc->SetOutput("output", {output_});
        op_desc->SetAttr<float>("scale_scale", scale_scale_);
        op_desc->SetAttr<float>("scale_bias", scale_bias_);
        op_desc->SetAttr<std::vector<int>>("transpose_axis", transpose_axis_);
    }

    void PrepareData() override {
        std::vector<float> data_q(dims_.production());
        fill_data_rand(data_q.data(), -1.f, 1.f, dims_.production());
        SetCommonTensor(input_q_, dims_, data_q.data());
    
        std::vector<float> data_k(dims_.production());
        fill_data_rand(data_k.data(), -1.f, 1.f, dims_.production());
        SetCommonTensor(input_k_, dims_, data_k.data());
    
        std::vector<float> data_v(dims_.production());
        fill_data_rand(data_v.data(), -1.f, 1.f, dims_.production());
        SetCommonTensor(input_v_, dims_, data_v.data());
    }
};

void test_sequence_qkv_attention(Place place) {
    int max_len = 10;
    DDim dims{{1, 2, 4, 3}};
    std::unique_ptr<arena::TestCase> tester(
        new QkvAttentionComputeTester(place, "def", dims));
    arena::Arena arena(std::move(tester), place, 2e-3);
    arena.TestPrecision();
}

TEST(XPUQkvAttention, precision) {
    paddle::lite::Place place;
#if defined(LITE_WITH_XPU)
    place = TARGET(kXPU);
#else
    return;
#endif
    test_sequence_qkv_attention(place);
}

}  // namespace lite
}  // namespace paddle



