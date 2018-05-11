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

#include "scope.h"
#include "tensor.h"
#include "variable.h"
#include "attribute.h"
#include "block_desc.h"
#include "common/types.h"
#include "common/variant.h"
#include "op_kernel_type.h"
#include "shape_inference.h"
#include "common/type_define.h"
#include "paddle_mobile_object.h"
#include "op_info.h"

namespace paddle_mobile {
namespace framework {
    /// If a variable is a empty variable, that name will be used.
    constexpr char kEmptyVarName[] = "@EMPTY@";

    /// If a variable is a temporary variable, that name will be set in Python,
    /// but it will be convert to a unique name in scope after OpCreator.
    constexpr char kTempVarName[] = "@TEMP@";

    template <typename Dtype>
    class ExecutionContext;

    template <typename Dtype>
    class OperatorBase: PaddleMobileObject {
    public:
        OperatorBase(const std::string& type, const VariableNameMap& inputs,
                     const VariableNameMap& outputs, const AttributeMap & attrs);
        virtual ~OperatorBase() {}

        template <typename T>
        inline const T& Attr(const std::string& name) const {
//            PADDLE_ENFORCE(attrs_.count(name) != 0, "%s should be in AttributeMap",
//                           name);
            return ((Attribute)attrs_.at(name)).Get<T>();
        }

        /// Net will call this interface function to Run an op.
        //  The implementation should be written at RunImpl
        virtual void Run(const Scope& scope);

        const VariableNameMap& Inputs() const { return inputs_; }
        const VariableNameMap& Outputs() const { return outputs_; }

        //! Get a input with argument's name described in `op_proto`
        std::string Input(const std::string& name) const;
        //! Get a input which has multiple variables.
        const std::vector<std::string>& Inputs(const std::string& name) const;

        std::vector<std::string> InputVars() const;

        //! Get a output with argument's name described in `op_proto`
        std::string Output(const std::string& name) const;
        //! Get an output which has multiple variables.
        //! TODO add a vector_view to prevent memory copy.
        const std::vector<std::string>& Outputs(const std::string& name) const;

        virtual std::vector<std::string> OutputVars(bool has_intermediate) const;

        const std::string& Type() const { return type_; }
        void SetType(const std::string& type) { type_ = type; }
        const AttributeMap& Attrs() const { return attrs_; }

        // Return a new operator instance, which is as same as this.
        // Use unique_ptr to prevent caller forget to delete this pointer.
//        virtual std::unique_ptr<OperatorBase> Clone() const = 0;

    protected:
        std::string type_;
        // NOTE: in case of OpGrad, inputs_ contains:
        // I (Inputs)
        // O (Outputs)
        // OG (Output Gradients)
        VariableNameMap inputs_;

        // NOTE: in case of OpGrad, outputs_ contains
        // IG (Inputs Gradients)
        VariableNameMap outputs_;
        AttributeMap attrs_;
    private:
        void GenerateTemporaryNames();
        void CheckAllInputOutputSet() const;
        virtual void RunImpl(const Scope& scope) const = 0;
    };

template <typename Dtype>
class OpKernelBase;

    template <typename Dtype>
    class OperatorWithKernel : public OperatorBase<Dtype>{
    public:
        using OpKernelMap =
        std::unordered_map<OpKernelType, std::unique_ptr<OpKernelBase<Dtype> >,
                OpKernelType::Hash>;

        OperatorWithKernel(const std::string& type, const VariableNameMap& inputs,
                           const VariableNameMap& outputs, const AttributeMap& attrs)
                : OperatorBase<Dtype>(type, inputs, outputs, attrs) {}

        static std::unordered_map<std::string /* op_type */, OpKernelMap>&
        AllOpKernels() {
            static std::unordered_map<std::string, OpKernelMap> g_all_op_kernels;
            return g_all_op_kernels;
        }

        virtual void InferShape(InferShapeContext* ctx) const {
            OpInfoMap<Dtype>::Instance().Get(OperatorBase<Dtype>::Type()).infer_shape_(ctx);
        }

    protected:
        virtual OpKernelType GetExpectedKernelType(const ExecutionContext<Dtype>& ctx) const;
        virtual OpKernelType GetKernelTypeForVar(
                const std::string& var_name, const Tensor& tensor,
                const OpKernelType& expected_kernel_type) const;
    private:
        // indicate kernel DataType by input data. By default all input data must be
        // same.
        proto::VarType::Type IndicateDataType(const ExecutionContext<Dtype>& ctx) const;
        void RunImpl(const Scope& scope) const final;
    };


    template <typename Dtype>
    class OpKernelBase: PaddleMobileObject{
    public:
        /**
         * ExecutionContext is the only parameter of Kernel Run function.
         * Run will get input/output variables, state such as momentum and
         * device resource such as CUDA stream, cublas handle, etc. from
         * ExecutionContext. User should construct it before run the Operator.
         */
        virtual void Compute(const ExecutionContext<Dtype>& context) const = 0;

        virtual ~OpKernelBase() = default;
    };


    template <typename Dtype>
    class OpKernel : public OpKernelBase<Dtype>{
    public:
    //        using ELEMENT_TYPE = T;
    };

    template <typename Dtype>
    class ExecutionContext {
    public:
        ExecutionContext(const OperatorBase<Dtype>& op, const Scope& scope)
                : op_(op), scope_(scope) {}

        const OperatorBase<Dtype>& op() const { return op_; }

        const Scope& scope() const { return scope_; }

        template <typename T>
        inline const T& Attr(const std::string& name) const {
            return op_.template Attr<T>(name);
        }

        size_t InputSize(const std::string& name) const {
            return op_.Inputs(name).size();
        }

        size_t OutputSize(const std::string& name) const {
            return op_.Outputs(name).size();
        }

        const Variable* InputVar(const std::string& name) const {
            auto ipt = op_.Input(name);
            return ipt == kEmptyVarName ? nullptr : scope_.FindVar(ipt);
        }

        Variable* OutputVar(const std::string& name) const {
            auto opt = op_.Output(name);
            return opt == kEmptyVarName ? nullptr : scope_.FindVar(opt);
        }

        const std::vector<const Variable*> MultiInputVar(
                const std::string& name) const {
            auto names = op_.Inputs(name);
            std::vector<const Variable*> res;
            res.reserve(names.size());
            std::transform(names.begin(), names.end(), std::back_inserter(res),
                           [this](const std::string& name) {
                               return name == kEmptyVarName ? nullptr
                                                            : scope_.FindVar(name);
                           });
            return res;
        }

        std::vector<Variable*> MultiOutputVar(const std::string& name) const {
            auto names = op_.Outputs(name);
            std::vector<Variable*> res;
            res.reserve(names.size());
            std::transform(names.begin(), names.end(), std::back_inserter(res),
                           [this](const std::string& name) {
                               return name == kEmptyVarName ? nullptr
                                                            : scope_.FindVar(name);
                           });
            return res;
        }

        template <typename T>
        const T* Input(const std::string& name) const {
            auto* var = InputVar(name);
            return var == nullptr ? nullptr : var->template Get<T>();
        }

        template <typename T>
        T* Output(const std::string& name) const {
            auto var = OutputVar(name);
            return var == nullptr ? nullptr : var->template GetMutable<T>();
        }

        template <typename T>
        const std::vector<const T*> MultiInput(const std::string& name) const {
            auto names = op_.Inputs(name);
            std::vector<const T*> res;
            res.reserve(names.size());
            std::transform(names.begin(), names.end(), std::back_inserter(res),
                           [&](const std::string& sub_name) {
                               auto var = scope_.FindVar(sub_name);
                               return var == nullptr ? nullptr : var->template Get<T>();
                           });
            return res;
        }

        template <typename T>
        std::vector<T*> MultiOutput(const std::string& name) const {
            auto names = op_.Outputs(name);
            std::vector<T*> res;
            res.reserve(names.size());
            std::transform(names.begin(), names.end(), std::back_inserter(res),
                           [&](const std::string& sub_name) {
                               auto var = scope_.FindVar(sub_name);
                               return var == nullptr ? nullptr : var->template GetMutable<T>();
                           });
            return res;
        }

        void ShareLoD(const std::string& in, const std::string& out, size_t i = 0,
                      size_t j = 0) const {
//            PADDLE_ENFORCE_LT(i, InputSize(in));
//            PADDLE_ENFORCE_LT(j, OutputSize(out));
            auto* in_var = MultiInputVar(in)[i];
            auto* out_var = MultiOutputVar(out)[j];
            if (!in_var->template IsType<LoDTensor>()) return;
//            PADDLE_ENFORCE(out_var->IsType<LoDTensor>(),
//                           "The %d-th output of Output(%s) must be LoDTensor.", j, out);
            auto in_tensor = in_var->template Get<LoDTensor>();
            auto* out_tensor = out_var->template GetMutable<LoDTensor>();
            out_tensor->set_lod(in_tensor->lod());
        }

        //! Get actual name vector for this input.
        const std::vector<std::string>& Inputs(const std::string& name) const {
            return op_.Inputs(name);
        }

        //! Get actual name vector for this output.
        const std::vector<std::string>& Outputs(const std::string& name) const {
            return op_.Outputs(name);
        }

    private:
        const OperatorBase<Dtype>& op_;
        const Scope& scope_;
    };

} // operators
} // paddle_mobile

