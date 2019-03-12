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

struct FlattenMetalParam {
    var idim: (Int32, Int32, Int32, Int32)
    var itrans: (Int32, Int32, Int32, Int32)
    var odim: (Int32, Int32, Int32, Int32)
    var otrans: (Int32, Int32, Int32, Int32)
}


class FlattenKernel<P: PrecisionProtocol>: Kernel, Computable{
    
    var metalParam: FlattenMetalParam
    
    required init(device: MTLDevice, param: FlattenParam<P>, initContext: InitContext) {
        param.output.initTexture(device: device, computePrecision: GlobalConfig.shared.computePrecision)
        var id: [Int32] = [1, 1, 1, 1]
        for i in 0..<param.input.tensorDim.cout() {
            id[4-param.input.tensorDim.cout()+i] = Int32(param.input.tensorDim[i])
        }
        let it: [Int32] = param.input.transpose.map { Int32($0) }
        var od: [Int32] = [1, 1, 1, 1]
        for i in 0..<param.output.tensorDim.cout() {
            od[4-param.output.tensorDim.cout()+i] = Int32(param.output.tensorDim[i])
        }
        let ot: [Int32] = param.output.transpose.map { Int32($0) }
        metalParam = FlattenMetalParam.init(
            idim: (id[0], id[1], id[2], id[3]),
            itrans: (it[0], it[1], it[2], it[3]),
            odim: (od[0], od[1], od[2], od[3]),
            otrans: (ot[0], ot[1], ot[2], ot[3])
        )
        let irank = param.input.tensorDim.cout()
        let orank = param.output.tensorDim.cout()
        assert(orank == 2)
        if GlobalConfig.shared.computePrecision == .Float32 {
            super.init(device: device, inFunctionName: "reshape_\(irank)_2_float", initContext: initContext)
        } else if GlobalConfig.shared.computePrecision == .Float16 {
            super.init(device: device, inFunctionName: "reshape_\(irank)_2_half", initContext: initContext)
        } else {
            fatalError()
        }
    }
    
    func compute(commandBuffer: MTLCommandBuffer, param: FlattenParam<P>) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PaddleMobileError.predictError(message: " encoder is nil")
        }
        
        encoder.setTexture(param.input.metalTexture, index: 0)
        encoder.setTexture(param.output.metalTexture, index: 1)
        
        encoder.setBytes(&metalParam, length: MemoryLayout<ReshapeMetalParam>.size, index: 0)
        encoder.dispatch(computePipline: pipline, outTexture: param.output.metalTexture)
        encoder.endEncoding()
    }
}
