// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <cmath>

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"

namespace nnadapter {
namespace qualcomm_qnn {
namespace htp {

/*
 * Relevant information on writing HTP op packages can be found in
 * "Op Writing Guidelines" section in QNN SDK docs/general/backend.html
 */
const char *op_name_relu = "Relu";
const char *op_name_relu_min_max = "ReluMinMax";
const char *op_name_relu_table_gen = "ReluTableGen";

using namespace hnnx;  // NOLINT
// op execute function declarations
// op 1
template <typename T_Ttype>
int ReluImpl(T_Ttype &out, const T_Ttype &in);  // NOLINT

// op 2
template <typename T_TtypeI, typename T_TtypeX>
int ReluMinMaxImpl(T_TtypeI &out,  // NOLINT
                   const T_TtypeI &in,
                   const T_TtypeX &in_x,
                   const T_TtypeX &in_y);

// op 3
GraphStatus ReluTablegenImpl(
    TensorContiguous<Tdefs::QuantUint8> &out,  // NOLINT
    const Tensor &in_step_size,
    const Tensor &in_offset,
    const Tensor &min,
    const Tensor &max);

}  // namespace htp
}  // namespace qualcomm_qnn
}  // namespace nnadapter

/*
 * op definitions
 * need to be global in the package
 * one definition per op
 */
using namespace nnadapter::qualcomm_qnn::htp;  // NOLINT

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and no
 flag
 * syntax: DEF_PACKAGE_OP(F,OP)
 */
DEF_PACKAGE_OP(ReluImpl<Tensor>, op_name_relu)
DEF_PACKAGE_OP((ReluMinMaxImpl<Tensor, Tensor>), op_name_relu_min_max)
DEF_PACKAGE_OP(ReluTablegenImpl, op_name_relu_table_gen)

/*
 * method 2 for defining op with specified cost value (one of "GLACIAL",
 * "SNAIL", "FAST", "FREE")
 * and provided flags
 * syntax: DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST,...)
 * can use zero (default) or more flags, FLAG options are IS_CONST,
 * INHIBIT_CONST_PROP,
 * RESOURCE_HVX, RESOURCE_HMX(not supported in external op packages)
 */
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((ReluImpl<PlainFloatTensor>),
                                  op_name_relu,
                                  "SNAIL",
                                  Flags::RESOURCE_HVX)

/*
 * method 3 for defining op with cost function pointer and provided flags
 * cost function pointer type: typedef float (*cost_function) (const Op *
 op);
 * syntax: DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST_F,...)
 */
DEF_PACKAGE_OP_AND_COST_AND_FLAGS(
    (ReluMinMaxImpl<QUint16CroutonTensor, Tensor>),
    op_name_relu_min_max,
    "SNAIL",
    Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS(
    (ReluMinMaxImpl<QUint16CroutonTensor_TCM, Tensor>),
    op_name_relu_min_max,
    "SNAIL",
    Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((ReluMinMaxImpl<QUint8CroutonTensor, Tensor>),
                                  op_name_relu_min_max,
                                  "SNAIL",
                                  Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS(
    (ReluMinMaxImpl<QUint8CroutonTensor_TCM, Tensor>),
    op_name_relu_min_max,
    "SNAIL",
    Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((ReluMinMaxImpl<QuantUint8Tensor, Tensor>),
                                  op_name_relu_min_max,
                                  "SNAIL",
                                  Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS(
    (ReluMinMaxImpl<QuantUint8Tensor_TCM, Tensor>),
    op_name_relu_min_max,
    "SNAIL",
    Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((ReluMinMaxImpl<PlainFloatTensor, Tensor>),
                                  op_name_relu_min_max,
                                  "SNAIL",
                                  Flags::RESOURCE_HVX)

/*
 * optimization definitions
 * need to be global in the package
 * one definition per optimization
 * syntax:
 * DEF_PACKAGE_OPTIMIZATION(PRIORITY,MATCHCODE,CONSTRAINTCODE,REPLACECODE)
 * PRIORITY predefined values include EARLY(2000), MIDDLE(3000), LATE(4000)
 * HTP core provides some replacement functions for op package to use
 * for more information about optimization rules, please refer to
 documentation
 *   located at QNN SDK docs/HTP/optimization_grammar.html
 */
DEF_PACKAGE_OPTIMIZATION(EARLY,
                         Op(op_name_relu, "X"),
                         IS_QUANT_TYPE("X"),
                         Op(op_name_relu_min_max,
                            "X",
                            gen_ConstScalar_f32(0.0f),
                            gen_ConstScalar_f32(INF)))

DEF_PACKAGE_OPTIMIZATION(EARLY,
                         Op("Relu6", "X"),
                         OK,
                         Op(op_name_relu_min_max,
                            "X",
                            gen_ConstScalar_f32(0.0f),
                            gen_ConstScalar_f32(6.0f)))

DEF_PACKAGE_OPTIMIZATION(EARLY,
                         Op("Relu1", "X"),
                         OK,
                         Op(op_name_relu_min_max,
                            "X",
                            gen_ConstScalar_f32(-1.0f),
                            gen_ConstScalar_f32(1.0f)))

DEF_PACKAGE_OPTIMIZATION(EARLY,
                         Op("ReluX", "X", "Max"),
                         OK,
                         Op(op_name_relu_min_max,
                            "X",
                            gen_ConstScalar_f32(0.0f),
                            gen_ConstScalar_f32(CONSTVAL_FLOAT("Max", 0))))

// Find min of range defined by a scale/offset for a qu8 tensor
#define MIN_QU8_CHECK(X) MUL(STEPSIZE_OF(X), MUL(-1.0f, ZERO_OFFSET_OF(X)))

// Find max of range defined by a scale/offset for a qu8 tensor
#define MAX_QU8_CHECK(X) MUL(STEPSIZE_OF(X), SUB(255.0f, ZERO_OFFSET_OF(X)))

// Drop relu if quant parms if input and output quant params indicate a range
of
    // min...max
    DEF_PACKAGE_OPTIMIZATION(
        EARLY,
        Op(op_name_relu_min_max, "In", "Min", "Max"),
        AND(IS_QUINT8("In"),
            IS_QUINT8("*"),
            EQ(STEPSIZE_OF("In"), STEPSIZE_OF("*")),
            EQ(ZERO_OFFSET_OF("In"), ZERO_OFFSET_OF("*")),
            GE(MIN_QU8_CHECK("In"), CONSTVAL_FLOAT("Min", 0)),
            LE(MAX_QU8_CHECK("In"), CONSTVAL_FLOAT("Max", 0))),
        "In")

        DEF_PACKAGE_OPTIMIZATION(
            EARLY + 1,
            Op(op_name_relu_min_max, "X", "Min", "Max"),
            AND(IS_QUINT8("X"), IS_QUINT8("*"), NOT(SAME_QUANT("X", "*"))),
            Op(FROM_DEFAULT_PACKAGE("TableLookup"),
               "X",
               WITH_SIZE(gen_Shape(1, 1, 1, 256),
                         Op(op_name_relu_table_gen,
                            gen_ConstScalar_f32(STEPSIZE_OF("X")),
                            gen_ConstScalar_i32(ZERO_OFFSET_OF("X")),
                            "Min",
                            "Max"))))

            DEF_PACKAGE_OPTIMIZATION(
                EARLY + 2,
                Op(op_name_relu_min_max, "X", "Min", "Max"),
                GT(DIM_BATCHES("*"), 1),
                AUTOSPLIT(0,
                          "I",
                          1,
                          Op(op_name_relu_min_max,
                             TYPICAL_SLICE("X", "I"),
                             "Min",
                             "Max")))

    // Split on depth
    DEF_PACKAGE_OPTIMIZATION(EARLY + 3,
                             Op(op_name_relu_min_max, "X", "Min", "Max"),
                             GT(DIM_DEPTH("*"), CHANNEL_SPLIT_SIZE),
                             AUTOSPLIT(3,
                                       "I",
                                       CHANNEL_SPLIT_SIZE,
                                       Op(op_name_relu_min_max,
                                          TYPICAL_SLICE("X", "I"),
                                          "Min",
                                          "Max")))

        DEF_PACKAGE_OPTIMIZATION(EARLY + 3,
                                 Op(op_name_relu, "X"),
                                 GT(DIM_DEPTH("*"), CHANNEL_SPLIT_SIZE),
                                 AUTOSPLIT(3,
                                           "I",
                                           CHANNEL_SPLIT_SIZE,
                                           Op(op_name_relu,
                                              TYPICAL_SLICE("X", "I"))))

    // Split on Height
    DEF_PACKAGE_OPTIMIZATION(EARLY + 4,
                             Op(op_name_relu_min_max, "X", "Min", "Max"),
                             GT(DIM_HEIGHT("*"), TILE_HEIGHT),
                             AUTOSPLIT(1,
                                       "I",
                                       TILE_HEIGHT,
                                       Op(op_name_relu_min_max,
                                          TYPICAL_SLICE("X", "I"),
                                          "Min",
                                          "Max")))

        DEF_PACKAGE_OPTIMIZATION(EARLY + 4,
                                 Op(op_name_relu, "X"),
                                 GT(DIM_HEIGHT("*"), TILE_HEIGHT),
                                 AUTOSPLIT(1,
                                           "I",
                                           TILE_HEIGHT,
                                           Op(op_name_relu,
                                              TYPICAL_SLICE("X", "I"))))

            DEF_PACKAGE_OPTIMIZATION(
                LATE + 10,
                Op(FROM_DEFAULT_PACKAGE("ConvLayer.opt.activations_to_vtcm"),
                   Op(op_name_relu_min_max, "X", "Min", "Max")),
                NOT(IS_FLOAT16("X")),
                Op(op_name_relu_min_max,
                   WITH_SAME_OUTPUT("X",
                                    Op(FROM_DEFAULT_PACKAGE(
                                           "ConvLayer.opt.activations_to_vtcm"),
                                       "X")),
                   "Min",
                   "Max"))

    // Do the below only for non-fp16 ReluMinMax. See ops_fp/src/fp16_relu.cc
    DEF_PACKAGE_OPTIMIZATION(
        LATE + 10,
        Op(op_name_relu_min_max,
           Op(FROM_DEFAULT_PACKAGE("ConvLayer.opt.activations_from_vtcm"), "X"),
           "Min",
           "Max"),
        NOT(IS_FLOAT16("X")),
        Op(FROM_DEFAULT_PACKAGE("ConvLayer.opt.activations_from_vtcm"),
           Op(op_name_relu_min_max, "X", "Min", "Max")))

        DEF_PACKAGE_OPTIMIZATION(
            LATE + 20,
            Op(op_name_relu_min_max,
               Op(FROM_DEFAULT_PACKAGE("flat_from_vtcm"), "X"),
               "Min",
               "Max"),
            NOT(IS_FLOAT16("X")),
            Op(FROM_DEFAULT_PACKAGE("flat_from_vtcm"),
               Op(op_name_relu_min_max, "X", "Min", "Max")))

    /*
     * op parameter order definitions
     * need to be global in the package
     * one definition per op, and this is optional
     * syntax:
     *
     DEF_PACKAGE_PARAM_ORDER(OP,PARAM1,MANDATORY1,DEFAULT1,PARAM2,MANDATORY2,DEFAULT2...)
     * one or more parameters can be specified for each op
     * order of parameters listed determines the order of parameters passed into
     op
     * execution functions
     * if an op does not have a parameter order definition, parameter order
     passed
     * into Qnn_addNode
     *   will be passed into op execution functions
     * if an op has a parameter order definition, any parameter passed into
     * Qnn_addNode with unlisted
     *   name will be abandoned
     * if two or more op packages with the same package name will be registered,
     * they cannot list
     *   conflicting parameter orders
     * PARAM refers to parameter name as a string literal
     * MANATORY refers to whether this parameter is required to be provided at
     * Qnn_addNode
     * DEFAULT is used when MANATORY is false
     *     if provided as Qnn_Param_t*,
     *       DEFAULT will be used for graph construction when this parameter is
     not
     * provided at
     *       Qnn_addNode
     *     if provided as nullptr,
     *       graph construction will skip this parameter when this parameter is
     not
     * provided at
     *       Qnn_addNode
     * eg. DEF_PACKAGE_PARAM_ORDER(op_name_relu,"X_VAL",true,nullptr)
     */
