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


#include "op_info.h"
#include "operator.h"
#include "var_type.h"
#include "selected_rows.h"
#include "data_transform.h"
#include "operators/conv_op.h"

namespace paddle_mobile {
namespace framework {

    template <typename Dtype>
    OperatorBase<Dtype>::OperatorBase(const std::string& type,
                               const VariableNameMap& inputs,
                               const VariableNameMap& outputs,
                               const AttributeMap& attrs)
            : type_(type), inputs_(inputs), outputs_(outputs), attrs_(attrs) {
        GenerateTemporaryNames();
        CheckAllInputOutputSet();
    }

    template <typename Dtype>
    void OperatorBase<Dtype>::Run(const Scope& scope) {
      RunImpl(scope);
    }

    template <typename Dtype>
    std::string OperatorBase<Dtype>::Input(const std::string& name) const {
        auto& ins = Inputs(name);
//        PADDLE_ENFORCE_LE(ins.size(), 1UL,
//                          "Operator %s's input %s should contain only one variable.",
//                          type_, name);
        return ins.empty() ? kEmptyVarName : ins[0];
    }

    template <typename Dtype>
    const std::vector<std::string>& OperatorBase<Dtype>::Inputs(
            const std::string& name) const {
        auto it = inputs_.find(name);
//        PADDLE_ENFORCE(it != inputs_.end(), "Operator %s does not have the input %s.",
//                       type_, name);
        return it->second;
    }

    template <typename Dtype>
    std::vector<std::string> OperatorBase<Dtype>::InputVars() const {
        std::vector<std::string> ret_val;
        for (auto& o : inputs_) {
            ret_val.reserve(ret_val.size() + o.second.size());
            ret_val.insert(ret_val.end(), o.second.begin(), o.second.end());
        }
        return ret_val;
    }

    template <typename Dtype>
    std::string OperatorBase<Dtype>::Output(const std::string& name) const {
        auto& outs = Outputs(name);
//        PADDLE_ENFORCE_LE(outs.size(), 1UL,
//                          "Operator %s's output %s should contain only one variable.",
//                          type_, name);
      for(auto &output : outs){
        std::cout << " out put: " << output << std::endl;
      }
        return outs.empty() ? kEmptyVarName : outs[0];
    }

    template <typename Dtype>
    const std::vector<std::string>& OperatorBase<Dtype>::Outputs(
            const std::string& name) const {
        auto it = outputs_.find(name);
//        PADDLE_ENFORCE(it != outputs_.end(),
//                       "Operator %s does not have an output called %s.", type_, name);
        return it->second;
    }

    template <typename Dtype>
    std::vector<std::string> OperatorBase<Dtype>::OutputVars(bool has_intermediate) const {
        std::vector<std::string> ret_val;
        if (has_intermediate) {
            // push all outputs into ret_val
            for (auto& o : outputs_) {
                ret_val.reserve(ret_val.size() + o.second.size());
                ret_val.insert(ret_val.end(), o.second.begin(), o.second.end());
            }
            return ret_val;
        }
        auto& info = OpInfoMap<Dtype>::Instance().Get(Type());

        // get all OpProto::Var for outputs
        for (auto& o : info.Proto().outputs()) {
            // ignore all intermediate output
            if (o.intermediate()) continue;
            auto out = outputs_.find(o.name());
            if (out != outputs_.end()) {
                ret_val.reserve(ret_val.size() + out->second.size());
                ret_val.insert(ret_val.end(), out->second.begin(), out->second.end());
            }
        }
        return ret_val;
    }

    template <typename Dtype>
    void OperatorBase<Dtype>::CheckAllInputOutputSet() const {
        auto& info_map = OpInfoMap<Dtype>::Instance();
        auto* op_info = info_map.GetNullable(Type());
        if (op_info == nullptr || op_info->proto_ == nullptr) return;

//        for (auto& in : op_info->Proto().inputs()) {
//            PADDLE_ENFORCE(inputs_.find(in.name()) != inputs_.end(),
//                           "Type %s's input %s is not set", Type(), in.name());
//        }
//
//        for (auto& out : op_info->Proto().outputs()) {
//            PADDLE_ENFORCE(outputs_.find(out.name()) != outputs_.end(),
//                           "Type %s's output %s is not set", Type(), out.name());
//        }
    }

    template <typename Dtype>
    void OperatorBase<Dtype>::GenerateTemporaryNames() {
        static std::atomic<size_t> gUniqId(0UL);
        for (auto& output : outputs_) {
            for (auto& output_name : output.second) {
                if (output_name == kTempVarName) {
                    output_name += type_;
                    output_name += "@";
                    output_name += std::to_string(gUniqId.fetch_add(1));
                }
            }
        }
    }


    static bool VarIsTensor(const Variable* var) {
        return var->IsType<LoDTensor>() || var->IsType<SelectedRows>();
    }

    static const Tensor* GetTensorFromVar(Variable* var) {
        if (var->IsType<LoDTensor>()) {
            return var->GetMutable<LoDTensor>();
        } else if (var->IsType<SelectedRows>()) {
            return var->GetMutable<SelectedRows>()->mutable_value();
        } else {
//            PADDLE_THROW("Variable type_id %s, expect LoDTensor/SelectedRows.",
//                         var->Type().name());
        }
    }

    static Tensor* GetMutableTensorFromVar(Variable* var) {
        if (var->IsType<LoDTensor>()) {
            return var->GetMutable<LoDTensor>();
        } else if (var->IsType<SelectedRows>()) {
            return var->GetMutable<SelectedRows>()->mutable_value();
        } else {
//            PADDLE_THROW("Variable type_id %s, expect LoDTensor/SelectedRows.",
//                         var->Type().name());
        }
    }

    template <typename Dtype>
    class RuntimeInferShapeContext : public InferShapeContext {
    public:
        RuntimeInferShapeContext(const OperatorBase<Dtype>& op, const Scope& scope)
                : op_(op), scope_(scope) {}

        bool HasInput(const std::string& name) const override {
            auto& ins = Inputs(name);
            size_t length = ins.size();
            if (length == 0) {
                return false;
            }
//            PADDLE_ENFORCE_EQ(length, 1UL,
//                              "Input %s should not have more than one inputs", name);
            auto ipt = ins[0];
            auto* var = ipt == kEmptyVarName ? nullptr : scope_.FindVar(ipt);
            return var != nullptr;
        }

        bool HasOutput(const std::string& name) const override {
            auto& outs = Outputs(name);
            size_t length = outs.size();
            if (length == 0) {
                return false;
            }
//            PADDLE_ENFORCE_EQ(length, 1UL,
//                              "Output %s should not have more than one inputs", name);
            auto ipt = outs[0];
            auto* var = ipt == kEmptyVarName ? nullptr : scope_.FindVar(ipt);
            return var != nullptr;
        }

        bool HasInputs(const std::string& name) const override {
            auto inputs = op_.Inputs(name);
            if (inputs.empty()) {
                return false;
            }
            for (auto& input : inputs) {
                if (scope_.FindVar(input) == nullptr) {
                    return false;
                }
            }
            return true;
        }

        bool HasOutputs(const std::string& name) const override {
            auto outputs = op_.Outputs(name);
            if (outputs.empty()) {
                return false;
            }
            for (auto& output : outputs) {
                if (scope_.FindVar(output) == nullptr) {
                    return false;
                }
            }
            return true;
        }

        AttrReader Attrs() const override { return AttrReader(op_.Attrs()); }

        const std::vector<std::string>& Inputs(
                const std::string& name) const override {
            return op_.Inputs(name);
        }

        const std::vector<std::string>& Outputs(
                const std::string& name) const override {
            return op_.Outputs(name);
        }

        void ShareLoD(const std::string& in, const std::string& out, size_t i = 0,
                      size_t j = 0) const override {
//            PADDLE_ENFORCE_LT(i, Inputs(in).size());
//            PADDLE_ENFORCE_LT(j, Outputs(out).size());
            Variable* in_var = scope_.FindVar(Inputs(in)[i]);
            Variable* out_var = scope_.FindVar(Outputs(out)[j]);
            if (!in_var->IsType<LoDTensor>()) return;
//            PADDLE_ENFORCE(out_var->IsType<LoDTensor>(),
//                           "The %d-th output of Output(%s) must be LoDTensor.", j, out);
            auto in_tensor = in_var->Get<LoDTensor>();
            auto* out_tensor = out_var->GetMutable<LoDTensor>();
            out_tensor->set_lod(in_tensor->lod());

            // TODO(dzhwinter) : reuse ShareLoD in most operators.
            // Need to call ShareLayout explicitly in sequence related ops.
            // Shall we have a better method to shared info between in/out Tensor?
            out_tensor->set_layout(in_tensor->layout());
        }

        void ShareLayout(const std::string& in, const std::string& out, size_t i = 0,
                         size_t j = 0) const {
//            PADDLE_ENFORCE_LT(i, Inputs(in).size());
//            PADDLE_ENFORCE_LT(j, Outputs(out).size());
            Variable* in_var = scope_.FindVar(Inputs(in)[i]);
            Variable* out_var = scope_.FindVar(Outputs(out)[j]);
            if (!in_var->IsType<LoDTensor>()) return;
//            PADDLE_ENFORCE(out_var->IsType<LoDTensor>(),
//                           "The %d-th output of Output(%s) must be LoDTensor.", j, out);
            auto in_tensor = in_var->Get<LoDTensor>();
            auto* out_tensor = out_var->GetMutable<LoDTensor>();
            out_tensor->set_layout(in_tensor->layout());
        }

        bool IsRuntime() const override { return true; }

    protected:
        DDim GetDim(const std::string& name) const override {
            Variable* var = scope_.FindVar(name);
            if (var->IsType<LoDTensor>()) {
                return var->Get<LoDTensor>()->dims();
            } else if (var->IsType<SelectedRows>()) {
                return var->Get<SelectedRows>()->GetCompleteDims();
            } else {
//                PADDLE_THROW(
//                        "Only LoDTensor/SelectedRows support 'GetDim', but Variable %s's "
//                                "type_id is %s.",
//                        name, var->Type().name());
            }
        }

        std::vector<DDim> GetRepeatedDims(const std::string& name) const override {
//            PADDLE_THROW("Only compile time support this method");
        }

        void SetDim(const std::string& name, const DDim& dim) override {
            Variable* var = scope_.FindVar(name);
            if (var->IsType<LoDTensor>()) {
                var->GetMutable<LoDTensor>()->Resize(dim);
            } else if (var->IsType<SelectedRows>()) {
                var->GetMutable<SelectedRows>()->set_height(dim[0]);
            } else {
//                PADDLE_THROW("Variable %s type_id %s, expect LoDTensor/SelectedRows.",
//                             name, var->Type().name());
            }
        }

        void SetRepeatedDims(const std::string& name,
                             const std::vector<DDim>& dims) override {
//            PADDLE_THROW("Only compile time support this method");
        }

        proto::VarType::Type GetVarType(const std::string& name) const override {
            auto* var = scope_.FindVar(name);
            return ToVarType(var->Type());
        }

        InferShapeVarPtr GetVarPtr(const std::string& name) override {
          InferShapeVarPtr ptr;
          ptr.Set<Variable*>(scope_.FindVar(name));
            return ptr;
        }

    private:
        const OperatorBase<Dtype>& op_;
        const Scope& scope_;
    };


    template <typename Dtype>
    void OperatorWithKernel<Dtype>::RunImpl(const Scope& scope) const {



        RuntimeInferShapeContext<Dtype> infer_shape_ctx(*this, scope);
        this->InferShape(&infer_shape_ctx);
//        platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
//        auto* dev_ctx = pool.Get(place);
//
//        // For profiling, don't move out of this function because that will result
//        // in the failure of multi-GPU profiling.
//        platform::RecordEvent record_event(Type(), dev_ctx);
        // check if op[type] has kernel registered.
        auto& all_op_kernels = AllOpKernels();
        auto kernels_iter = all_op_kernels.find(OperatorBase<Dtype>::type_);
//        if (kernels_iter == all_op_kernels.end()) {
//            PADDLE_THROW(
//                    "There are no kernels which are registered in the %s operator.", type_);
//        }




        ExecutionContext<Dtype> ctx(*this, scope);

//        OpKernelMap& kernels = kernels_iter->second;

        // TODO(dzhwinter) : kernel fallback mechanism will be added when all the
        // transform functions are ready.

        // for (auto& candidate : kKernelPriority) {
        //   Do selection
        // }


//      const ExecutionContext<Dtype>& ctx1 = ctx;
      std::cout << " in run impl GetExpectedKernelType " << std::endl;
//      auto expected_kernel_key = this->GetExpectedKernelType(ctx);
      std::cout << " out run impl GetExpectedKernelType " << std::endl;




//        VLOG(3) << "expected_kernel_key:" << expected_kernel_key;

//      auto kernel_iter = kernels.find(expected_kernel_key);



      operators::GemmConvKernel<Dtype, float> kernel;


//        if (kernel_iter == kernels.end()) {
//            PADDLE_THROW("op %s does not have kernel for %s", type_,
//                         KernelTypeToString(expected_kernel_key));
//        }

        // do data transform
//        Scope& new_scope = scope.NewScope();
//
//        std::vector<std::string> inplace_vars;
//        for (auto& var_name_item : this->Inputs()) {
//            for (auto& var_name : var_name_item.second) {
//                auto* var = scope.FindVar(var_name);
//                if (var && VarIsTensor(var)) {
//                    auto* tensor_in = GetTensorFromVar(var);
//                    if (tensor_in->IsInitialized()) {
//                        auto kernel_type_for_var = this->GetKernelTypeForVar(
//                                var_name_item.first, *tensor_in, expected_kernel_key);
//                        if (TransFromNeeded(kernel_type_for_var, expected_kernel_key)) {
//                            auto out_var_names = OperatorBase<Dtype>::OutputVars(true);
//                            if (std::find(out_var_names.begin(), out_var_names.end(),
//                                          var_name) != out_var_names.end()) {
//                                inplace_vars.push_back(var_name);
//                            }
////                            VLOG(3) << "Transform Variable " << var_name << " from "
////                                    << kernel_type_for_var << " to " << expected_kernel_key;
//                            auto* trans_var = new_scope.Var(var_name);
//                            std::shared_ptr<Tensor> out(new Tensor);
//                            DataTransform(expected_kernel_key, kernel_type_for_var, *tensor_in,
//                                          out.get());
//                            CopyVariableWithTensor(*var, *(out.get()), *trans_var);
//                        }
//                    }
//                }
//            }
//        }


      kernel.Compute(
                ExecutionContext<Dtype>(*this, scope));
//
//        for (auto& var_name : inplace_vars) {
////            VLOG(3) << "share inplace var " + var_name + " back to it's original scope";
//            auto* original_tensor = GetMutableTensorFromVar(scope.FindVar(var_name));
//            auto* transformed_tensor = GetTensorFromVar(new_scope.FindVar(var_name));
//            original_tensor->ShareDataWith(*transformed_tensor);
//        }

    }

    template <typename Dtype>
    proto::VarType::Type OperatorWithKernel<Dtype>::IndicateDataType(
            const ExecutionContext<Dtype>& ctx) const {
        std::cout << " begin IndicateDataType " << std::endl;
        auto& scope = ctx.scope();
        int data_type = -1;
        for (auto& input : this->inputs_) {
            for (auto& ipt_name : input.second) {
                auto* var = scope.FindVar(ipt_name);
                if (var != nullptr) {
                    const Tensor* t = nullptr;
                    if (var->template IsType<Tensor>()) {
                        t = var->template Get<Tensor>();
                    } else if (var->template IsType<LoDTensor>()) {
                        t = var->template Get<LoDTensor>();
                    } else if (var->template IsType<SelectedRows>()) {
                        t = &(var->template Get<SelectedRows>()->value());
                    }
                    if (t != nullptr) {
                        int tmp = static_cast<int>(ToDataType(t->type()));
//                        PADDLE_ENFORCE(tmp == data_type || data_type == -1,
//                                       "DataType of Paddle Op %s must be the same.", Type());
                        data_type = tmp;
                    }
                }
            }
        }
      std::cout << " end IndicateDataType " << std::endl;
//        PADDLE_ENFORCE(data_type != -1, "DataType should be indicated by input");
        return static_cast<proto::VarType::Type>(data_type);
    }

    template <typename Dtype>
    OpKernelType OperatorWithKernel<Dtype>::GetExpectedKernelType(
            const ExecutionContext<Dtype>& ctx) const {
//    std::cout << " in GetExpectedKernelType " << std::endl;
//    printf("in GetExpectedKernelType");
//    throw std::bad_exception();

      return OpKernelType(IndicateDataType(ctx));
    }

    template <typename Dtype>
    OpKernelType OperatorWithKernel<Dtype>::GetKernelTypeForVar(
            const std::string& var_name, const Tensor& tensor,
            const OpKernelType& expected_kernel_type) const {
        return OpKernelType(expected_kernel_type.data_type_);
    }

template class OpKernel<ARM>;
template class OpKernelBase<ARM>;
template class OperatorBase<ARM>;
template class OperatorWithKernel<ARM>;

} // framework
} // paddle_mobile
