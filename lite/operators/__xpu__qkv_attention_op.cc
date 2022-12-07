/*
* user: by wangrui39
* data: 2022.10.31
*/
#include <vector>

#include "lite/operators/__xpu__qkv_attention_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XPUQkvAttentionOp::CheckShape() const {
    param_.input_k->dims().size();
    CHECK_EQ_OR_FALSE(param_.input_k->dims().size(), 4);
    CHECK_EQ_OR_FALSE(param_.input_q->dims().size(),  param_.input_k->dims().size());
    CHECK_EQ_OR_FALSE(param_.input_v->dims().size(),  param_.input_k->dims().size());

    // Only the 3 input dims of q, k and v are equal
    for (size_t i = 0; i < 4; i++) {
        CHECK_EQ_OR_FALSE(param_.input_k->dims()[i], param_.input_q->dims()[i]);
        CHECK_EQ_OR_FALSE(param_.input_q->dims()[i], param_.input_v->dims()[i]);
    }
    return true;
}
 
bool XPUQkvAttentionOp::InferShapeImpl() const {
    param_.output->Resize(param_.input_v->dims());
    return true;
}

bool XPUQkvAttentionOp::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
    auto q = op_desc.Input("input_q").front();
    auto k = op_desc.Input("input_k").front();
    auto v = op_desc.Input("input_v").front();
    auto out = op_desc.Output("output").front();

    param_.input_k = scope->FindVar(q)->GetMutable<lite::Tensor>();
    param_.input_q = scope->FindVar(k)->GetMutable<lite::Tensor>();
    param_.input_v = scope->FindVar(v)->GetMutable<lite::Tensor>();
    param_.output = scope->FindVar(out)->GetMutable<lite::Tensor>();
    return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__qkv_attention, paddle::lite::operators::XPUQkvAttentionOp);
