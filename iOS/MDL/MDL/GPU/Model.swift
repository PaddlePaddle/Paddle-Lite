/* Copyright (c) 2017 Baidu, Inc. All Rights Reserved.
 
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

import Foundation

import MetalPerformanceShaders

class ParamModel {
    private(set) var output_num: Int = 0
    private(set) var kernel_size: Int = 0
    private(set) var pad: Int = 0
    private(set) var stride: Int = 0
    private(set) var type: String?
    private(set) var global_pooling: Bool = false
    init(values: NSDictionary){
        for key in values.allKeys where key is NSString {
            let sKey = key as! NSString
            if sKey.isEqual(to: "output_num") {
                output_num = values[sKey] as? Int ?? 0
            }else if sKey.isEqual(to: "kernel_size"){
                kernel_size = values[sKey] as? Int ?? 0
            }else if sKey.isEqual(to: "pad"){
                pad = values[sKey] as? Int ?? 0
            }else if sKey.isEqual(to: "stride"){
                stride = values[sKey] as? Int ?? 0
            }else if sKey.isEqual(to: "type"){
                type = values[sKey] as? String
            }else if sKey.isEqual(to: "global_pooling"){
                global_pooling = values[sKey] as? Bool ?? false
            }
        }
    }
}

@available(iOS 10.0, *)
class LayerModel {
    static let convolutionType = "ConvolutionLayer"
    static let concatType = "ConcatLayer"
    static let fcType = "FCLayer"
    static let poolType = "PoolingLayer"
    static let reluType = "ReluLayer"
    static let splitType = "SplitLayer"
    static let depthWiseConvolutionType = "DepthwiseConvolutionLayer"
    static let softmaxType = "SoftmaxLayer"
    static let denseType = "Dense"
    static let customType = "Custom"
    static let resizeType = "Resize"
    static let pointWiseType = "PointwiseConvolutionLayer"
    static let averagePoolingType = "AveragePooling"
    static let globalAveragePoolingType = "GlobalAveragePooling"
    static let activationType = "Activation"
    static let maxPool = "MaxPool"
    static let preProcessType = "PreProcess"
    
    var inputMatrices: [Matrix] = []
    var outputMatrices: [Matrix] = []
    var weightMatrices: [Matrix] = []
    
    var pid: Int?
    var neurons: Int?
    var name: String = ""
    var type: String = ""
    var input: [String] = []
    var output: [String] = []
    var weight: [String] = []
    var param: ParamModel?
    var relu: Bool = false
    var reluA: Float = 0.0
    var nextLayer: [LayerModel]?
    var destinationChannelOffset: Int = 0

    init(values: NSDictionary) {
        for key in values.allKeys {
            guard let sKey = key as? NSString else {
                break
            }
            
            if sKey == "name"{
                name = values[sKey] as? String ?? ""
            }else if sKey == "type"{
                type = values[sKey] as? String ?? ""
            }else if sKey.isEqual(to: "input"){
                input = values[sKey] as? [String] ?? []
            }else if sKey.isEqual(to: "output"){
                output = values[sKey] as? [String] ?? []
            }else if sKey.isEqual(to: "weight"){
                weight = values[sKey] as? [String] ?? []
            }else if sKey.isEqual(to: "param"){
                guard let paraDic = values["param"] as? NSDictionary else {
                    break
                }
                param = ParamModel(values: paraDic)
            }else if sKey.isEqual(to: "pid"){
                pid = values[sKey] as? Int
            }else if sKey.isEqual(to: "relu"){
                relu = values[sKey] as? Bool ?? true
            }else if sKey.isEqual(to: "reluA"){
                reluA = values[sKey] as? Float ?? 0.0
            } else if sKey.isEqual(to: "neurons"){
                neurons = values[sKey] as? Int
            }
        }
    }
    init() {
    }
}

@available(iOS 10.0, *)
class Model{
    internal(set) var layer: [LayerModel] = []
    internal(set) var matrix: [String : [Int]] = [:]
    var matrixDesDic: [Matrix : MPSImageDescriptor] = [:]

    init(values: NSDictionary) {
        for key in values.allKeys{
            if let sKey = key as? String{
                if sKey == "layer"{
                    guard let layerArr = values[sKey] as? NSArray else {
                        break
                    }
                    layer.removeAll()
                    for layerDic in layerArr where layerDic is NSDictionary{
                        let layerModel = LayerModel(values: layerDic as! NSDictionary)
                        layer.append(layerModel)
                    }
                }else if sKey == "matrix"{
                    if let tMatrixs = values[sKey] as? [String : [Int]]{
                        matrix = tMatrixs
                    }
                }
            }
        }
    }
}
