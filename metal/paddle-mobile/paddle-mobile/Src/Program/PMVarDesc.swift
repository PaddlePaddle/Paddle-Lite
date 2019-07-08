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

public enum VarTypeType: Int {
    case ErrorType = -1,
    Bool = 0,
    Int16 = 1,
    Int32 = 2,
    Int64 = 3,
    FP16 = 4,
    FP32 = 5,
    FP64 = 6,
    LodTensor = 7,
    SelectedRows = 8,
    FeedMiniBatch = 9,
    FetchList = 10,
    StepScopes = 11,
    StepLodRankTable = 12,
    StepLodTensorArray = 13,
    StepPlaceList = 14,
    Reader = 15,
    Channel = 16,
    Raw = 17,
    Tuple = 18
    
    func dataTypeSize() throws -> Int {
        switch self {
        case .FP16:
            return 2
        case .FP32:
            return 4
        case .FP64:
            return 8
        case .Int32:
            return 4
        case .Int64:
            return 8
        case .Bool:
            return 1
        default:
            throw PaddleMobileError.makeError(type: .memoryError, msg: "not support \(self) type to get size")
        }
    }
}

public class PMVarDesc {
    public let name: String
    public let persistable: Bool
    public let type: VarTypeType
    let tensorDesc: TensorDesc?
    public var dims: [Int]? {
        get {
            return tensorDesc?.dims
        }
    }
    init(protoVarDesc: VarDesc) {
        type = VarTypeType.init(rawValue: Int(protoVarDesc.type.type.rawValue)) ?? .ErrorType
        name = protoVarDesc.name
        persistable = protoVarDesc.persistable
        switch type {
        case .SelectedRows:
            tensorDesc = TensorDesc.init(protoTensorDesc: protoVarDesc.type.selectedRows)
        case .LodTensor:
            tensorDesc = TensorDesc.init(protoTensorDesc: protoVarDesc.type.lodTensor.tensor)
        case .StepLodTensorArray:
            tensorDesc = TensorDesc.init(protoTensorDesc: protoVarDesc.type.tensorArray_p.tensor);
        default:
            tensorDesc = .none
        }
    }
}

extension PMVarDesc: CustomStringConvertible, CustomDebugStringConvertible {
    public var description: String {
        var str = ""
        str += "var name \(name): \n"
        if let inTensorDesc = tensorDesc {
            str += " dim size: \(inTensorDesc.dims.count) \n"
            str += "    dim: \(inTensorDesc.dims) \n"
            str += "type:\(self.type) \n"
        } else {
            str += " no dim info"
        }
        
        return str
    }
    
    public var debugDescription: String {
        return description
    }
}
