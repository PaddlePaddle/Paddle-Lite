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

@available(iOS 10.0, *)
class Loader {
    static let share:Loader = Loader()
    private let stringSize: CInt = 30
    var matrices: [String : Matrix] = [:]
    var layerModelDic: [String : LayerModel] = [:]
    
    private init(){}
    
    func clear() {
        matrices.removeAll()
        layerModelDic.removeAll()
    }
    
    func load(device: MTLDevice, modelPath: String, para: (String, Int) -> NetParameterData?) throws -> Model{
        return try loadMatrix(device: device, modelPath: modelPath) { (name, size) -> NetParameterData? in
            return para(name, size)
        }
    }
    
    func load(device: MTLDevice, modelPath: String, weightPath: String) throws -> Model   {
        let model = try loadMatrix(device: device, modelPath: modelPath, para: { (name, size) -> NetParameterData? in
            return  ParameterData(size: size)
        })
        try loadBinary(weightPath: weightPath)
        return model
    }
    
    private func loadMatrix(device: MTLDevice, modelPath: String, para: (String, Int) -> NetParameterData?) throws -> Model{
        let fileManager = FileManager.default
        let data = fileManager.contents(atPath: modelPath)
        
        guard let d = data else {
            throw NetError.loaderError(message: "has no data in the filePath: " + modelPath)
        }
        var jsonStr = try String.init(contentsOfFile: modelPath, encoding: .utf8)
        if jsonStr.contains("="){
            var index = jsonStr.index(before: jsonStr.endIndex)
            while jsonStr[index] != Character.init("}"){
                index = jsonStr.index(before: index)
            }
            jsonStr.removeSubrange(jsonStr.index(after: index)..<jsonStr.endIndex)
        }
        
        guard let jsonData = jsonStr.data(using: .utf8) else {
            throw NetError.loaderError(message: "json string -> json data error, json structure error")
        }
        
        let dic = try JSONSerialization.jsonObject(with: jsonData, options: .allowFragments)
        let model = try convert(originModel: Model(values: dic as! NSDictionary))
        
        func createMatrix(name: String) throws -> Matrix{
            guard let matrixConfig = model.matrix[name] else {
                throw NetError.loaderError(message: "matrix in json lack " + name)
            }
            let matrix = Matrix(device: device, name: name, config: matrixConfig)
            return matrix
        }
        for (index, layer) in model.layer.enumerated(){
            if index == 0 {
                for input in layer.input{
                    let matrix = try createMatrix(name: input)
                    layer.inputMatrices.append(matrix)
                }
            }else{
                if layer.type == LayerModel.concatType {
                    var concatConfig = [0, 0, 0, 0]
                    for input in layer.input {
                        let config = model.matrix[input]
                        concatConfig[0] = config?[0] ?? 0
                        concatConfig[1] += config?[1] ?? 0
                        concatConfig[2] = config?[2] ?? 0
                        concatConfig[3] = config?[3] ?? 0
                    }
                    let matrix = Matrix(device: device, name: layer.name, config: concatConfig)
                    model.matrixDesDic[matrix] = matrix.createImageDes()
                    layer.inputMatrices = [matrix]
                    layer.outputMatrices = [matrix]
                    matrices[layer.output[0]] = matrix
                    
                    
                    var channels: Int = 0
                    for input in layer.input {
                        let previousLayerModel = layerModelDic[input]
                        previousLayerModel?.destinationChannelOffset = channels

                        let previousMatrix = matrices[input]
                        if let inPreMatrix = previousMatrix{
                            model.matrixDesDic[inPreMatrix] = nil
                        }
                        
                        channels += previousMatrix?.channels ?? 0
                        previousLayerModel?.outputMatrices.first?.concatMatrix = matrix
                    }
                    
                }else{
                    for input in layer.input {
                        guard let inputMatrix = matrices[input] else {
                            throw NetError.loaderError(message: "no input matrix in " + layer.name)
                        }
                        
                        layer.inputMatrices.append(inputMatrix)
                    }
                }
            }
            if layer.type != LayerModel.concatType{
                for output in layer.output{
                    let matrix = try createMatrix(name: output)
                    matrices[output] = matrix
                    layerModelDic[output] = layer
                    layer.outputMatrices.append(matrix)
                    model.matrixDesDic[matrix] = matrix.createImageDes()
                }
                for weigth in layer.weight{
                    let matrix = try createMatrix(name: weigth)
                    matrices[weigth] = matrix
                    matrix.data = para(matrix.name, matrix.count())
                }
            }
        }
        return model
    }

    private func loadBinary(weightPath: String) throws{
        let dataGetter = try FileDataGetter<CInt>(filePath: weightPath)
        var binaryData = dataGetter.data
        let modelVersion = binaryData[1]
        let modelCount = binaryData[2]
        
        print("\(modelVersion) --- \(modelCount)")
        binaryData = binaryData.advanced(by: 3)
        
        var modelSizes: [CInt] = []
        for i in 0..<modelCount {
            modelSizes.append(binaryData[i])
        }
        binaryData = binaryData.advanced(by: Int(modelCount))
        
        var floatBinaryData:UnsafeMutablePointer<CFloat>  = unsafeBitCast(binaryData, to: UnsafeMutablePointer<CFloat>.self)
        var modelMins: [CFloat] = []
        var modelMaxs: [CFloat] = []
        for i in 0..<modelCount {
            modelMins.append(floatBinaryData[i * 2])
            modelMaxs.append(floatBinaryData[i * 2 + 1])
        }
        floatBinaryData = floatBinaryData.advanced(by: Int(modelCount) * 2)
        
        var charBinaryData: UnsafeMutablePointer<CChar> = unsafeBitCast(floatBinaryData, to: UnsafeMutablePointer<CChar>.self)
        var tempCharBinaryData = charBinaryData
        
        var modelNames: [String] = []
        for _ in 0..<modelCount {
            let modelName = String.init(cString: tempCharBinaryData)
            modelNames.append(modelName)
            tempCharBinaryData = tempCharBinaryData.advanced(by: Int(stringSize))
        }
        charBinaryData = charBinaryData.advanced(by: Int(modelCount * stringSize))
        
        var uInt8Data: UnsafeMutablePointer<UInt8> = unsafeBitCast(charBinaryData, to: UnsafeMutablePointer<UInt8>.self)
        
        for i in 0..<modelCount {
            let modelSize = modelSizes[i]
            let modelName = modelNames[i]
            
            if modelName == "data"{
                continue
            }
            
            guard let matrix = matrices[modelName] else {
                throw NetError.loaderError(message: "can't find " + modelName + " in matrixs when load binary file")
            }
            
            guard matrix.count() == Int(modelSize) else {
                throw NetError.loaderError(message: "matrix count does not match between json and binary file")
            }
            let minvalue = modelMins[i]
            
            let maxValue = modelMaxs[i]
            let factor: CFloat = (maxValue - minvalue)/255.0
            guard var matrixData = matrix.data?.pointer else{
                throw NetError.loaderError(message: "matrix: " + modelName + " data pointer hasn't initialized")
            }
            
            for _ in 0..<modelSize/4{
                matrixData[0] = CFloat(uInt8Data[0]) * factor + minvalue
                matrixData[1] = CFloat(uInt8Data[1]) * factor + minvalue
                matrixData[2] = CFloat(uInt8Data[2]) * factor + minvalue
                matrixData[3] = CFloat(uInt8Data[3]) * factor + minvalue
                matrixData = matrixData.advanced(by: 4)
                uInt8Data = uInt8Data.advanced(by: 4)
            }
            for j in 0..<modelSize%4{
                matrixData[j] = CFloat(uInt8Data[j]) * factor + minvalue
            }
            uInt8Data = uInt8Data.advanced(by: Int(modelSize%4))
        }
    }
    
    private func convert(originModel: Model) throws -> Model{
        let resultModel = Model(values: [:])
        resultModel.matrix = originModel.matrix
        resultModel.layer = []
        var layerDic: [String : LayerModel] = [:]
        for layer in originModel.layer {
            if let output = layer.output.first, layer.type != LayerModel.reluType{
                layerDic[output] = layer
            }
            if layer.type == LayerModel.reluType {
                guard let input = layer.input.first else {
                    throw NetError.modelDataError(message:"input of " + layer.name + " has no element")
                }
                guard let inputLayer = layerDic[input] else {
                    throw NetError.modelDataError(message: "")
                }
                inputLayer.relu = true
                
            }else{
                resultModel.layer.append(layer)
            }
        }
        return resultModel
    }
}
