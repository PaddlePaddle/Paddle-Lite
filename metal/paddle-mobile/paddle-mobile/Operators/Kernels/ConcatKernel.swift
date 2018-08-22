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

import Foundation

struct ConcatTestParam: TestParam {
    var input: [MTLTexture]
    var output: MTLTexture
    var dims: [[Int]]
    var axis: Int
    var odim: [Int]
}

struct ConcatMetalParam {
    var odim: (Int32, Int32, Int32, Int32) = (1, 1, 1, 1)
    var axis: Int32 = 0
    var offset: Int32 = 0
    var vdim: (Int32, Int32, Int32, Int32, Int32, Int32) = (0, 0, 0, 0, 0, 0)
}

class ConcatKernel<P: PrecisionType>: Kernel, Computable{
    
    func encodeTest(_ cmdBuffer: MTLCommandBuffer, _ param: ConcatTestParam, _ istart: Int, _ iend: Int) {
        let encoder = cmdBuffer.makeComputeCommandEncoder()!
        var p = ConcatMetalParam.init()
        var odim: [Int32] = [1, 1, 1, 1]
        for i in 0..<param.odim.count {
            odim[4-param.odim.count+i] = Int32(param.odim[i])
        }
        p.odim = (odim[0], odim[1], odim[2], odim[3])
        p.axis = Int32(4 - param.odim.count + param.axis)
        for i in 0..<istart {
            p.offset += Int32(param.dims[i][param.axis])
        }
        var vdim: [Int32] = []
        for i in 0..<(iend - istart) {
            encoder.setTexture(param.input[i+istart], index: i)
            vdim.append(Int32(param.dims[i+istart][Int(param.axis)]))
        }
        for i in (iend-istart)..<6 {
            encoder.setTexture(param.input[0], index: i)
            vdim.append(0)
        }
        p.vdim = (vdim[0], vdim[1], vdim[2], vdim[3], vdim[4], vdim[5])
        encoder.setTexture(param.output, index: 6)
        encoder.setTexture(param.output, index: 7)
        encoder.setBytes(&p, length: MemoryLayout<ConcatMetalParam>.size, index: 0)
        encoder.dispatch(computePipline: pipline, outTexture: param.output)
        encoder.endEncoding()
    }
    
    func encode(_ cmdBuffer: MTLCommandBuffer, _ param: ConcatParam<P>, _ istart: Int, _ iend: Int) throws {
        guard let encoder = cmdBuffer.makeComputeCommandEncoder() else {
            throw PaddleMobileError.predictError(message: " encode is nil")
        }
        var p = ConcatMetalParam.init()
        let odim = (0..<4).map { Int32(param.output.dim[$0]) }
        p.odim = (odim[0], odim[1], odim[2], odim[3])
        p.axis = Int32(4 - param.output.tensorDim.cout() + param.axis)
        for i in 0..<istart {
            p.offset += Int32(param.input[i+istart].dim[Int(p.axis)])
        }
        var vdim: [Int32] = []
        for i in 0..<(iend - istart) {
            encoder.setTexture(param.input[i+istart].metalTexture, index: i)
            vdim.append(Int32(param.input[i+istart].dim[Int(p.axis)]))
        }
        for i in (iend-istart)..<6 {
            encoder.setTexture(param.input[0].metalTexture, index: i)
            vdim.append(0)
        }
        p.vdim = (vdim[0], vdim[1], vdim[2], vdim[3], vdim[4], vdim[5])
        encoder.setTexture(param.output.metalTexture, index: 6)
        encoder.setTexture(param.output.metalTexture, index: 7)
        encoder.setBytes(&p, length: MemoryLayout<ConcatMetalParam>.size, index: 0)
        encoder.dispatch(computePipline: pipline, outTexture: param.output.metalTexture)
        encoder.endEncoding()
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: ConcatParam<P>) throws {
        for i in 0..<param.input.count {
            for j in 0..<4 {
                assert(param.input[i].transpose[j] == j)
            }
        }
        
        let group = param.input.count / 6
        let remain = param.input.count % 6
        for i in 0..<group {
            try self.encode(commandBuffer, param, 6 * i, 6 * (i + 1))
        }
        if remain > 0 {
            try self.encode(commandBuffer, param, 6 * group, param.input.count)
        }
    }
    
    func test(cmdBuffer: MTLCommandBuffer, param: ConcatTestParam) {
        let group = param.input.count / 6
        let remain = param.input.count % 6
        for i in 0..<group {
            try self.encodeTest(cmdBuffer, param, 6 * i, 6 * (i + 1))
        }
        if remain > 0 {
            try self.encodeTest(cmdBuffer, param, 6 * group, param.input.count)
        }
    }
    
    required init(device: MTLDevice, param: ConcatParam<P>) {
        param.output.initTexture(device: device)
        super.init(device: device, inFunctionName: "concat")
    }
    
    required init(device: MTLDevice, testParam: ConcatTestParam) {
        super.init(device: device, inFunctionName: "concat")
    }
}
