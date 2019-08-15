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

import Metal
import Foundation

public class PaddleMobileUnitTest {
    let device: MTLDevice
    let queue: MTLCommandQueue
    public init(inDevice: MTLDevice, inQueue: MTLCommandQueue) {
        device = inDevice
        queue = inQueue
    }
    
    private func indentPrintTensor(tensor: [Float32], dim: [Int], ix: [Int], indentLevel: Int) {
        let indent = Array.init(repeating: " ", count: indentLevel).joined(separator: "")
        var tx = ix
        if dim.count == indentLevel + 1 {
            var log: String = indent + "["
            for i in 0..<dim[indentLevel] {
                tx = ix
                tx[indentLevel] = i
                for x in 1..<dim.count {
                    for y in 0..<x {
                        tx[y] *= dim[x]
                    }
                }
                let c = tx.reduce(0) { $0 + $1 }
                if i > 0 {
                    log += ", "
                }
                log += tensor[c].description
            }
            log += "]"
            if (indentLevel > 0) && (ix[indentLevel - 1] < dim[indentLevel - 1] - 1) {
                log += ","
            }
            print(log)
        } else {
            print(indent + "[")
            for i in 0..<dim[indentLevel] {
                tx[indentLevel] = i
                indentPrintTensor(tensor: tensor, dim: dim, ix: tx, indentLevel: indentLevel + 1)
            }
            if (indentLevel > 0) && (ix[indentLevel - 1] < dim[indentLevel - 1] - 1) {
                print(indent + "],")
            } else {
                print(indent + "]")
            }
        }
    }
    
    private func tensorPrint(tensor: [Float32], dim: [Int]) {
        var detectPos = -1
        var odim = 1
        var ndim = dim
        for i in 0..<dim.count {
            if dim[i] == -1 {
                if detectPos == -1 {
                    detectPos = i
                } else {
                    detectPos = -2
                }
            } else if dim[i] <= 0 {
                detectPos = -3
            } else {
                odim *= dim[i]
            }
        }
        guard detectPos >= -1 else {
            print("must satisfy detectPos >= -1")
            return
        }
        if (detectPos == -1) {
            guard tensor.count == odim else {
                print("must satisfy tensor.count == odim")
                return
            }
        } else {
            guard tensor.count % odim == 0 else {
                print("must satisfy tensor.count % odim == 0")
                return
            }
            ndim[detectPos] = tensor.count / odim
        }
        indentPrintTensor(tensor: tensor, dim: ndim, ix: dim.map { $0 * 0 }, indentLevel: 0)
    }
    
    public func testConcat() {
        //        let buffer = queue.makeCommandBuffer() ?! "buffer is nil"
        //        var it: [[Float32]] = []
        //        for _ in 0..<7 {
        //            it.append((0..<12).map { Float32($0) })
        //        }
        //        let input = it.map { device.tensor2texture(value: $0, dim: [3, 4]) }
        //        let output = device.tensor2texture(value: [Float32](), dim: [3, 28])
        //
        //        let param = ConcatTestParam.init(
        //            input: input,
        //            output: output,
        //            dims: [[3, 4], [3, 4], [3, 4], [3, 4], [3, 4], [3, 4], [3, 4]],
        //            axis: 1,
        //            odim: [3, 28]
        //        )
        //        let concatKernel = ConcatKernel<Float32>.init(device: device, testParam: param)
        //        concatKernel.test(cmdBuffer: buffer, param: param)
        //        buffer.addCompletedHandler { (buffer) in
        //            for i in 0..<it.count {
        //                let _: Float32? = input[i].logDesc()
        //                self.tensorPrint(tensor: it[i], dim: [3, 4])
        //            }
        //            let _: Float32? = output.logDesc()
        //            let tx: [Float32] = self.device.texture2tensor(texture: output, dim: [3, 28])
        //            self.tensorPrint(tensor: tx, dim: [3, 28])
        //        }
        //
        //        buffer.commit()
    }
    
    public func testReshape() {
        //        let buffer = queue.makeCommandBuffer() ?! "buffer is nil"
        //        let input: [Float32] = (0..<24).map { Float32($0) }
        //        let inTexture = device.tensor2texture(value: input, dim: [2, 3, 4])
        //        let outTexture = device.tensor2texture(value: [Float32](), dim: [4, 6])
        //        let mp = ReshapeMetalParam.init(
        //            idim: (1, 2, 3, 4),
        //            itrans: (0, 1, 2, 3),
        //            odim: (1, 1, 4, 6),
        //            otrans: (0, 1, 2, 3)
        //        )
        //        let param = ReshapeTestParam.init(
        //            inputTexture: inTexture,
        //            outputTexture: outTexture,
        //            param: mp
        //        )
        //        let reshapeKernel = ReshapeKernel<Float32>.init(device: device, testParam: param)
        //        reshapeKernel.test(commandBuffer: buffer, testParam: param)
        //        buffer.addCompletedHandler { (buffer) in
        //            let _: Float32? = inTexture.logDesc()
        //            let _: Float32? = outTexture.logDesc()
        //            self.tensorPrint(tensor: input, dim: [2, 3, 4])
        //            let tx: [Float32] = self.device.texture2tensor(texture: outTexture, dim: [4, 6])
        //            self.tensorPrint(tensor: tx, dim: [4, 6])
        //        }
        
        //        let input: [Float32] = (0..<24).map { Float32($0) }
        //        let inTexture = device.tensor2texture(value: input, dim: [2, 3, 4])
        //        let outTexture = device.tensor2texture(value: [Float32](), dim: [24])
        //        let mp = ReshapeMetalParam.init(
        //            idim: (1, 2, 3, 4),
        //            itrans: (0, 1, 2, 3),
        //            odim: (1, 1, 1, 24),
        //            otrans: (0, 1, 2, 3)
        //        )
        //        let param = ReshapeTestParam.init(
        //            inputTexture: inTexture,
        //            outputTexture: outTexture,
        //            param: mp
        //        )
        //        let reshapeKernel = ReshapeKernel<Float32>.init(device: device, testParam: param)
        //        reshapeKernel.test(commandBuffer: buffer, testParam: param)
        //        buffer.addCompletedHandler { (buffer) in
        //            let _: Float32? = inTexture.logDesc()
        //            let _: Float32? = outTexture.logDesc()
        //            self.tensorPrint(tensor: input, dim: [2, 3, 4])
        //            let tx: [Float32] = self.device.texture2tensor(texture: outTexture, dim: [24])
        //            self.tensorPrint(tensor: tx, dim: [24])
        //        }
        //
        //        
        //        buffer.commit()
    }
    
    public func testTranspose() {
        
        guard let buffer = queue.makeCommandBuffer() else {
            return
        }
        //        var input: [Float32] = []
        //        for i in 0..<72 {
        //            input.append(Float32(i))
        //        }
        ////        let inputTexture = device.makeFloatTexture(value: input, textureWidth: 3, textureHeight: 2, arrayLength: 3)
        //        let inputTexture = device.tensor2texture(value: input, dim: [4, 3, 2, 3]);
        //        // group 1
        //        let outputTexture = device.tensor2texture(value: [Float32](), dim: [3, 3, 2, 4])
        //        let param = TransposeTestParam.init(inputTexture: inputTexture, outputTexture: outputTexture, iC: 3, oC: 4, axis: [3, 1, 2, 0])
        ////        let param = TransposeTestParam.init(inputTexture: inputTexture, outputTexture: outputTexture, iC: 4, oC: 2, axis: [3, 0, 2, 1])
        ////        // group 2
        ////        let outputTexture = device.makeFloatTexture(value: [Float32](), textureWidth: 3, textureHeight: 3, arrayLength: 6)
        ////        let param = TransposeTestParam.init(inputTexture: inputTexture, outputTexture: outputTexture, iC: 4, oC: 4, axis: [3, 0, 2, 1])
        ////
        //        let transposeKernel = TransposeKernel<Float32>.init(device: device, testParam: param)
        //
        //        transposeKernel.test(commandBuffer: buffer, param: param)
        //
        //        buffer.addCompletedHandler { (buffer) in
        //            let _: Float32? = inputTexture.logDesc(header: "input texture", stridable: false)
        //            let _: Float32? = outputTexture.logDesc(header: "output texture", stridable: false)
        //            self.tensorPrint(tensor: input, dim: [4, 3, 2, 3])
        //            let tx: [Float32] = self.device.texture2tensor(texture: outputTexture, dim: [3, 3, 2, 4])
        //            self.tensorPrint(tensor: tx, dim: [3, 3, 2, 4])
        //        }
        //
        //        let input: [Float32] = (0..<24).map { Float32($0) }
        //        let inputTexture = device.tensor2texture(value: input, dim: [2, 3, 4])
        //        let outputTexture = device.tensor2texture(value: [Float](), dim: [3, 4, 2])
        //        let param = TransposeTestParam.init(inputTexture: inputTexture, outputTexture: outputTexture, iC: 4, oC: 2, axis: [0, 2, 3, 1])
        //        let transposeKernel = TransposeKernel<Float32>.init(device: device, testParam: param)
        //
        //        transposeKernel.test(commandBuffer: buffer, param: param)
        //
        //        buffer.addCompletedHandler { (buffer) in
        //            let _: Float32? = inputTexture.logDesc(header: "input texture", stridable: false)
        //            let _: Float32? = outputTexture.logDesc(header: "output texture", stridable: false)
        //            self.tensorPrint(tensor: input, dim: [2, 3, 4])
        //            let tx: [Float32] = self.device.texture2tensor(texture: outputTexture, dim: [3, 4, 2])
        //            self.tensorPrint(tensor: tx, dim: [3, 4, 2])
        //        }
        //        
        buffer.commit()
    }
    
    public func testConvAddBnRelu() {
        guard let buffer = queue.makeCommandBuffer() else {
            return
        }
        
        let input: [Float32] = [
            1.0, 2.0, 3.0, 4.0,
            1.0, 2.0, 3.0, 4.0,
            1.0, 2.0, 3.0, 4.0,
            
            1.0, 2.0, 3.0, 4.0,
            1.0, 2.0, 3.0, 4.0,
            1.0, 2.0, 3.0, 4.0,
            
            1.0, 2.0, 3.0, 4.0,
            1.0, 2.0, 3.0, 4.0,
            1.0, 2.0, 3.0, 4.0,
            ]
        
        let filter: [Float32] = [
            //1.0
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            
            //2.0
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            
            //3.0
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            
            //4.0
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            ]
        
        let biase: [Float32] = [1.0, 1.0, 1.0, 100.0]
        let newScalue: [Float32] = [1.0, 1.0, 1.0, 1.0]
        let newBiase: [Float32] = [1.0, 1.0, 1.0, 1.0]
        
        let inputeTexture = device.makeFloatTexture(value: input, textureWidth: 3, textureHeight: 3, arrayLength: 1)
        
        //filter
        let filterBuffer = try! device.makeBuffer(value: filter)
        
        // biase
        let biaseBuffer = try! device.makeBuffer(value: biase)
        
        // new scale
        let newScalueBuffer = try! device.makeBuffer(value: newScalue)
        
        // new biase
        let newBiaseBuffer = try! device.makeBuffer(value: newBiase)
        
        //output
        let outputTexture = device.makeFloatTexture(value: [Float32](), textureWidth: 2, textureHeight: 2, arrayLength: 1)
        
        let filterSize: (width: Int, height: Int, channel: Int) = (3, 3, 4)
        let paddings: (Int, Int) = (1, 1)
        let stride: (Int, Int) = (2, 2)
        
        let offsetX = filterSize.width/2 - paddings.0
        let offsetY = filterSize.height/2 - paddings.1
        
        let groups = 1
        let iC = 4
        let fC = 4
        let oC = 4
        
        let metalParam = MetalConvParam.init(offsetX: Int16(offsetX), offsetY: Int16(offsetY), offsetZ: 0, strideX: UInt16(stride.0), strideY: UInt16(stride.1), dilationX: UInt16(1), dilationY: UInt16(1), groups: UInt16(groups), iC: UInt16(iC), fC: UInt16(fC), oC: UInt16(oC), hasAddOp: UInt16(0), hasReluOp: UInt16(0), addParam: ElementwiseAddMetalParam())
        
        let param = ConvAddBatchNormReluTestParam.init(inInputTexture: inputeTexture, inOutputTexture: outputTexture, inMetalParam: metalParam, inFilterBuffer: filterBuffer, inBiaseBuffer: biaseBuffer, inNewScaleBuffer: newScalueBuffer, inNewBiaseBuffer: newBiaseBuffer, inFilterSize: filterSize)
        
        let initContext = InitContext.init()
        initContext.metalLoadMode = .LoadMetalInDefaultLib
        
        let convAddBnReluKernel = try! ConvAddBatchNormReluKernel<Float32>.init(device: device, testParam: param, initContext: initContext)
        
        try! convAddBnReluKernel.test(commandBuffer: buffer, param: param)
        
        buffer.addCompletedHandler { (buffer) in
            let _: Float32? = inputeTexture.logDesc(header: "input texture", stridable: false)
            let _: Float32? = outputTexture.logDesc(header: "output texture", stridable: false)
        }
        
        buffer.commit()
    }
}



