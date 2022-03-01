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

#include "driver/intel_openvino/utility.h"


namespace nnadapter {
namespace intel_openvino {

PadType ConvertToOVPadType(NNAdapterAutoPadCode& auto_pad_code) {
    switch (auto_pad_code) {
        case NNADAPTER_AUTO_PAD_VALID:
            return PadType::VALID;
        case NNADAPTER_AUTO_PAD_SAME:
            return PadType::SAME_UPPER;
        case NNADAPTER_AUTO_PAD_NONE:
            return PadType::NOTSET;
        default:
            return PadType::NOTSET;
    }
}

ElementType ConvertToOVElementType(NNAdapterOperandPrecisionCode& precision_code) {
    std::map<NNAdapterOperandPrecisionCode, ElementType> precision_map {
        {NNADAPTER_BOOL8, ov::element::boolean},
        {NNADAPTER_INT8, ov::element::i8}, 
        {NNADAPTER_QUANT_INT8_SYMM_PER_LAYER, ov::element::i8}, 
        {NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL, ov::element::i8}, 
        {NNADAPTER_UINT8, ov::element::u8}, 
        {NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER, ov::element::u8}, 
        {NNADAPTER_INT16, ov::element::i16}, 
        {NNADAPTER_QUANT_INT16_SYMM_PER_LAYER, ov::element::i16}, 
        {NNADAPTER_QUANT_INT16_SYMM_PER_CHANNEL, ov::element::i16}, 
        {NNADAPTER_INT32, ov::element::i32}, 
        {NNADAPTER_QUANT_INT32_SYMM_PER_LAYER, ov::element::i32}, 
        {NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL, ov::element::i32}, 
        {NNADAPTER_UINT32, ov::element::u32}, 
        {NNADAPTER_QUANT_UINT32_ASYMM_PER_LAYER, ov::element::u32}, 
        {NNADAPTER_INT64, ov::element::i64}, 
        {NNADAPTER_UINT64, ov::element::u64}, 
        {NNADAPTER_FLOAT16, ov::element::f16},
        {NNADAPTER_FLOAT32, ov::element::f32},
        {NNADAPTER_FLOAT64, ov::element::f64}
    };
    auto it = precision_map.find(precision_code);
    if (it != precision_map.end()) {
        return it->second;
    }
    NNADAPTER_LOG(FATAL)
                << "Failed to convert the NNAdapter operand precision code("
                << OperandPrecisionCodeToString(precision_code)
                << ") to OpenVINO's element type !";
    return ov::element::f32;
}
    
#define FUNCTION_ADD_CONST_OUTPUT_NODE_DEFINE(type, element_type) \
    template<> std::shared_ptr<OutputNode> AddConstOutputNode<type> (std::vector<size_t> dimensions, std::vector<type> values) { \
        auto const_node = std::make_shared<default_opset::Constant>(element_type,  Shape(dimensions), values); \
        return std::make_shared<OutputNode>(const_node->output(0)); \
    }
FUNCTION_ADD_CONST_OUTPUT_NODE_DEFINE(int8_t, ov::element::i8)
FUNCTION_ADD_CONST_OUTPUT_NODE_DEFINE(int16_t, ov::element::i16)
FUNCTION_ADD_CONST_OUTPUT_NODE_DEFINE(int32_t, ov::element::i32)
FUNCTION_ADD_CONST_OUTPUT_NODE_DEFINE(int64_t, ov::element::i64)
FUNCTION_ADD_CONST_OUTPUT_NODE_DEFINE(uint8_t, ov::element::u8)
FUNCTION_ADD_CONST_OUTPUT_NODE_DEFINE(uint16_t, ov::element::u16)
FUNCTION_ADD_CONST_OUTPUT_NODE_DEFINE(uint32_t, ov::element::u32)
FUNCTION_ADD_CONST_OUTPUT_NODE_DEFINE(uint64_t, ov::element::u64)
FUNCTION_ADD_CONST_OUTPUT_NODE_DEFINE(float, ov::element::f32)
FUNCTION_ADD_CONST_OUTPUT_NODE_DEFINE(double, ov::element::f64)
#undef FUNCTION_ADD_CONST_OUTPUT_NODE_DEFINE

} // namespace intel_openvino
} // namespace nnadapter