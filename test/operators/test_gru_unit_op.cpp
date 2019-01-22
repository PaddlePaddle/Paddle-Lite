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

#include "../test_include.h"
#include "operators/gru_unit_op.h"

namespace paddle_mobile {

int TesGruUnitOp(const std::vector<int> input_shape, int axis) {
	framework::DDim x_dims = framework::make_ddim(input_shape);
	framework::DDim norm_dims = x_dims;
	if (axis < 0) {
		axis += x_dims.size();
	}
	norm_dims[axis] = 1;

	VariableNameMap inputs;
	VariableNameMap outputs;
	auto scope = std::make_shared<framework::Scope>();
	inputs["X"] = std::vector<std::string>({"input"});
	outputs["Norm"] = std::vector<std::string>({"outputNorm"});
	outputs["Out"] = std::vector<std::string>({"output"});

	auto input_var = scope.get()->Var("input");
	auto input = input_var->template GetMutable<framework::LoDTensor>();
	SetupTensor<float>(input, x_dims, -100.0, 100.0);

	auto norm_var = scope.get()->Var("outputNorm");
	auto norm = norm_var->template GetMutable<framework::LoDTensor>();
	SetupTensor<float>(norm, norm_dims, -100.0, 100.0);

	auto output_var = scope.get()->Var("output");
	framework::AttributeMap attrs;
	auto *op = new operators::GruUnitOp<CPU, float>("norm", inputs, outputs, attrs, scope);

	op->InferShape();
	op->Init();
	op->Run();
}

} // namespace paddle_mobile

int main() {

}
