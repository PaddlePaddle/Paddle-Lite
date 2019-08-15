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

public enum PaddleMobileErrorType {
    case loaderError
    case netError
    case memoryError
    case paramError
    case opError
    case predictError
    case defaultError
}

public enum PaddleMobileError: Error{
    case loaderError(message: String)
    case netError(message: String)
    case memoryError(message: String)
    case paramError(message: String)
    case opError(message: String)
    case predictError(message: String)
    case defaultError(message: String)
    
    static public func makeError(type: PaddleMobileErrorType, msg: String, file: String = #file, line: Int = #line, function: String = #function, callStack: Array<String> = Thread.callStackSymbols) -> PaddleMobileError {
        paddleMobileLog(msg, logLevel: .FatalError, file: file, line: line, function: function, callStack: callStack)
        let debugMsg = "\(msg) -file: \(file) -line: \(line) -function:\(function) -calling stack: \(callStack)"
        switch type {
        case .loaderError:
            return PaddleMobileError.loaderError(message: debugMsg)
        case .netError:
            return PaddleMobileError.netError(message: debugMsg)
        case .memoryError:
            return PaddleMobileError.memoryError(message: debugMsg)
        case .paramError:
            return PaddleMobileError.paramError(message: debugMsg)
        case .opError:
            return PaddleMobileError.opError(message: debugMsg)
        case .predictError:
            return PaddleMobileError.predictError(message: debugMsg)
        case .defaultError:
            return PaddleMobileError.defaultError(message: debugMsg)
        }
    }
}
