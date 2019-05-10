//
//  ConvAddReluKernel.swift
//  paddle-mobile
//
//  Created by Yang,Yanzhan on 2019/4/29.
//  Copyright © 2019 orange. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class ConvAddReluKernel<P: PrecisionProtocol>: ConvAddKernel<P> {
    override class func kernelFunctionName(param: ConvAddParam<P>, useAggressiveOptimization: Bool = false) -> String? {
        if GlobalConfig.shared.computePrecision == .Float16 {
            if param.filter.width == 1 && param.filter.height == 1 {
                return "conv_add_relu_1x1_half"
            } else if param.filter.channel == 1 && param.filter.n == param.input.tensorDim[1] {
                if useAggressiveOptimization {
                    let couldUseWinograd = param.filter.width == 3 && param.filter.height == 3
                        && param.filter.n == 16 && param.stride[0] == 1 && param.stride[1] == 1
                        && param.dilations[0] == 1 && param.dilations[1] == 1
                    if couldUseWinograd {
                        return "depthwise_conv_add_relu_3x3_half_winograd"
                    }
                }
                return "depthwise_conv_add_relu_3x3_half"
            } else if param.filter.width == 3 && param.filter.height == 3 {
                return "conv_add_relu_3x3_half"
            } else if param.filter.width == 1 && param.filter.height == 5 {
                return "conv_add_relu_5x1_half"
            } else if param.filter.width == 5 && param.filter.height == 1 {
                return "conv_add_relu_1x5_half"
            } else {
                return nil
            }
        } else if GlobalConfig.shared.computePrecision == .Float32 {
            if param.filter.width == 1 && param.filter.height == 1 {
                return "conv_add_relu_1x1"
            } else if param.filter.channel == 1 && param.filter.n == param.input.tensorDim[1] {
                return "depthwise_conv_add_relu_3x3"
            } else if param.filter.width == 1 && param.filter.height == 5 {
                return "conv_add_relu_5x1"
            } else if param.filter.width == 5 && param.filter.height == 1 {
                return "conv_add_relu_1x5"
            } else if param.filter.width == 3 && param.filter.height == 3 {
                return "conv_add_relu_3x3"
            } else {
                return nil
            }
        } else {
            return nil
        }
    }
    
    override func neuronFilterForMPSLayer(device: MTLDevice) -> AnyObject? {
        if #available(iOS 10.0, *) {
            return MPSCNNNeuronReLU(device: device, a: 0)
        }
        return nil
    }
}
