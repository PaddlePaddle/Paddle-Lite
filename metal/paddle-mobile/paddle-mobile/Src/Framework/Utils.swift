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

enum PaddleMobileLogLevel: String {
    case Info = "Info"
    case Warning = "Warning"
    case FatalError = "FatalError"
}

func paddleMobileLog(_ msg: String, logLevel: PaddleMobileLogLevel = .Info, file: String = #file, line: Int = #line, function: String = #function, callStack: Array<String>? = nil) {
    var msg = "PaddleMobileLog-\(logLevel.rawValue): \(msg) -file: \(file) -line: \(line) -function:\(function)"
    if let callStack = callStack {
        msg.append(contentsOf: " -calling stack: \(callStack)")
    }
    print(msg)
}

func paddleMobileThrow(error: PaddleMobileError, file: String = #file, line: Int = #line, function: String = #function, callStack: Array<String> = Thread.callStackSymbols) -> PaddleMobileError {
    let debugMsg = "-file: \(file) -line: \(line) -function:\(function) -calling stack: \(callStack)"
    switch error {
    case .loaderError(message: let errorMsg):
        let newMsg = "\(errorMsg) \(debugMsg)"
        return PaddleMobileError.loaderError(message: newMsg)
    case .netError(message: let errorMsg):
        let newMsg = "\(errorMsg) \(debugMsg)"
        return PaddleMobileError.netError(message: newMsg)
    case .memoryError(message: let errorMsg):
        let newMsg = "\(errorMsg) \(debugMsg)"
        return PaddleMobileError.memoryError(message: newMsg)
    case .paramError(message: let errorMsg):
        let newMsg = "\(errorMsg) \(debugMsg)"
        return PaddleMobileError.paramError(message: newMsg)
    case .opError(message: let errorMsg):
        let newMsg = "\(errorMsg) \(debugMsg)"
        return PaddleMobileError.opError(message: newMsg)
    case .predictError(message: let errorMsg):
        let newMsg = "\(errorMsg) \(debugMsg)"
        return PaddleMobileError.predictError(message: newMsg)
    case .defaultError(message: let errorMsg):
        let newMsg = "\(errorMsg) \(debugMsg)"
        return PaddleMobileError.defaultError(message: newMsg)
    }
}

func paddleMobileLogAndThrow(error: PaddleMobileError, file: String = #file, line: Int = #line, function: String = #function, callStack: Array<String> = Thread.callStackSymbols) -> PaddleMobileError {
    paddleMobileLog(error.associatedMsg(), logLevel: .FatalError, file: file, line: line, function: function, callStack: callStack)
    return paddleMobileThrow(error: error, file: file, line: line, function: function, callStack: callStack)
}
