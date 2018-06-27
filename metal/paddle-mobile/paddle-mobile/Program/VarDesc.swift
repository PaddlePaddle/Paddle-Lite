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

enum VarTypeType: Int {
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
            throw PaddleMobileError.memoryError(message: "not support \(self) type to get size ")
        }
    }
}

struct VarDesc {
    let name: String
    let persistable: Bool
    let type: VarTypeType
    let tensorDesc: TensorDesc?
    init(protoVarDesc: PaddleMobile_Framework_Proto_VarDesc) {
        type = VarTypeType.init(rawValue: protoVarDesc.type.type.rawValue) ?? .ErrorType
        name = protoVarDesc.name
        persistable = protoVarDesc.persistable
        switch type {
        case .SelectedRows:
            tensorDesc = TensorDesc.init(protoTensorDesc: protoVarDesc.type.selectedRows)
        case .LodTensor:
            tensorDesc = TensorDesc.init(protoTensorDesc: protoVarDesc.type.lodTensor.tensor)
        case .StepLodTensorArray:
            tensorDesc = TensorDesc.init(protoTensorDesc: protoVarDesc.type.tensorArray.tensor);
        default:
            tensorDesc = .none
        }
    }
    
}
