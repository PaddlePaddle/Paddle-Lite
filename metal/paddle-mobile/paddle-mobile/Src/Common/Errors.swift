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

public enum PaddleMobileError: Error{
    case loaderError(message: String)
    case netError(message: String)
    case memoryError(message: String)
    case paramError(message: String)
    case opError(message: String)
    case predictError(message: String)
    case defaultError(message: String)
    
    func associatedMsg() -> String {
        switch self {
        case .loaderError(message: let msg):
            return msg
        case .netError(message: let msg):
            return msg
        case .memoryError(message: let msg):
            return msg
        case .paramError(message: let msg):
            return msg
        case .opError(message: let msg):
            return msg
        case .predictError(message: let msg):
            return msg
        case .defaultError(message: let msg):
            return msg
        }
    }
}
