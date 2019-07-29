//
//  TestViewController.swift
//  PaddleMobileTest
//
//  Created by Li,Jian(MMS) on 2019/6/27.
//  Copyright © 2019 Li,Jian(MMS). All rights reserved.
//

import UIKit
import Alamofire
import paddle_mobile

class Model: CustomStringConvertible {
    var name = ""
    var paramsPrecision: [Int] = [0, 1]
    var fusion: [Bool] = [true, false]
    var useMPS: [Bool] = [true, false]
    var reuseTexture: [Bool] = [true, false]
    var testPerformance = true
    var diffPrecision: Float = 0.1
    var varsDic: [String: [Int]] = [:]
    
    var description: String {
        return "name: \(name)\nfusion: \(fusion)\nuseMPS:\(useMPS)\nreuseTexture:\(reuseTexture)\ntestPerformance: \(testPerformance)\ndiffPrecision: \(diffPrecision)\nvarsDesc:\(varsDic)"
    }
}

class TestModel: CustomStringConvertible, Codable {
    var name: String
    var paramsPrecision: Int
    var fusion: Bool
    var useMPS: Bool
    var reuseTexture: Bool
    var testPerformance: Bool
    var diffPrecision: Float
    var varsDic: [String: [Int]]
    
    init(name: String, paramsPrecision: Int, fusion: Bool, useMPS: Bool, reuseTexture: Bool, testPerformance: Bool, diffPrecision: Float, varsDic: [String: [Int]]) {
        self.name = name
        self.paramsPrecision = paramsPrecision
        self.fusion = fusion
        self.useMPS = useMPS
        self.reuseTexture = reuseTexture
        self.testPerformance = testPerformance
        self.diffPrecision = diffPrecision
        self.varsDic = varsDic
    }
    
    var description: String {
        return "name: \(name)\nparamsPrecision:\(paramsPrecision)\nfusion: \(fusion)\nuseMPS:\(useMPS)\nreuseTexture:\(reuseTexture)\ntestPerformance: \(testPerformance)\ndiffPrecision: \(diffPrecision)\nvarsDesc:\(varsDic)"
    }
}

class ModelTestResult: Codable {
    var model: TestModel?
    var isResultEqual = false
    var performance = 0.0
    var msg = ""
    var diverVarName = ""
}

let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!
var timeCosts = [Double]()
var count = 0
var totalCount = 100

var orderedVars: [String] = []
var varIndex = 0

class TestViewController: UIViewController {
    private var hostUrlStr = ""
    private var testModels: [TestModel] = []
    private var testResults: [ModelTestResult] = []
    private var runner: Runner?
    private var textView: UITextView!
    
    init(hostUrlStr: String) {
        //self.hostUrlStr = "https://www.baidu.com"//
        self.hostUrlStr = hostUrlStr
        super.init(nibName: nil, bundle: nil)
        self.title = self.hostUrlStr
    }
    
    required init?(coder aDecoder: NSCoder) {
        return nil
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        textView = UITextView(frame: self.view.bounds)
        self.view.addSubview(textView)
        textView.text = "hello"
        textView.backgroundColor = .white
        textView.isEditable = false
        
        let button1 = UIButton(frame: CGRect(x: 100, y: 100, width: 100, height: 50))
        button1.center = CGPoint(x: self.view.center.x, y: 200) 
        button1.addTarget(self, action: #selector(test1), for: .touchUpInside)
        button1.backgroundColor = .lightGray
        button1.setTitle("开始测试", for: .normal)
        self.view.addSubview(button1)
    }
    
    @objc func test1() {
        getTestInfo()
    }
    
    private func getTestInfo() {
        var testInfoRequest = URLRequest(url: URL(string: "\(hostUrlStr)/getTestInfo")!)
        testInfoRequest.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        Alamofire.request(testInfoRequest).validate().responseJSON { (response) in
            guard response.result.isSuccess else {
                self.testLog("getTestInfo request error")
                return
            }
            guard let json = response.result.value as? [String: Any] else {
                self.testLog("getTestInfo serialize error")
                return
            }
            if let modelList = json["model_list"] as? [[String: Any]] {
                self.testModels.removeAll()
                for model in modelList {
                    guard let name = model["name"] as? String else {
                        self.testLog("getTestInfo fetch name error")
                        return
                    }
                    guard let varsDic = model["vars_dic"] as? [String: [Int]] else {
                        self.testLog("getTestInfo fetch vars_dic error")
                        return
                    }
                    let paddlemodel = Model()
                    paddlemodel.name = name
                    paddlemodel.varsDic = varsDic
                    if let paramsPrecision = model["params_precision"] as? [Int] {
                        paddlemodel.paramsPrecision = paramsPrecision
                    }
                    if let fusion = model["fusion"] as? [Bool] {
                        paddlemodel.fusion = fusion
                    }
                    if let useMPS = model["use_mps"] as? [Bool] {
                        paddlemodel.useMPS = useMPS
                    }
                    if let reuseTexture = model["reuse_texture"] as? [Bool] {
                        paddlemodel.reuseTexture = reuseTexture
                    }
                    if let testPerformance = model["test_performance"] as? NSNumber {
                        paddlemodel.testPerformance = testPerformance.boolValue
                    }
                    if let diffPrecision = model["diff_precision"] as? NSNumber {
                        paddlemodel.diffPrecision = diffPrecision.floatValue
                    }
                    self.testModels.append(contentsOf: self.testModelsFromModel(paddlemodel))
                }
                if !self.testModels.isEmpty {
                    self.beginTest()
                }
            }
        }
    }
    
    private func beginTest() {
        testResults.removeAll()
        testModelOfIndex(0)
    }
    
    private func testModelOfIndex(_ index: Int) {
        guard index < self.testModels.count else {
            //report
            self.testLog("********test result: done")
            let jsonEncoder = JSONEncoder()
            let jsonData = try! jsonEncoder.encode(["results": self.testResults])
            let json = String(data: jsonData, encoding: String.Encoding.utf8) ?? ""
            let data = json.data(using: .utf8)!
            let jsonDic = try! JSONSerialization.jsonObject(with: data, options: []) as? [String: Any]
            Alamofire.request("\(hostUrlStr)/putTestResult", method: .post, parameters: jsonDic, encoding: JSONEncoding.default, headers: nil).validate().responseData { (response) in
                guard response.result.isSuccess else {
                    self.testLog("puttestresult fail\ntest done")
                    return
                }
                self.testLog("puttestresult success\test done")
            }
            return
        }
        let model = self.testModels[index]
        self.testLog("******begin test model: \(model)")
        testModel(model) { [weak self] (testResult) in
            self?.testLog("********test result: isresultequal: \(testResult.isResultEqual) msg: \(testResult.msg) performance: \(testResult.performance) s diverVar: \(testResult.diverVarName)")
            self?.testResults.append(testResult)
            self?.testModelOfIndex(index+1)
        }
    }
    
    private func testModel(_ model: TestModel, completion: @escaping ((ModelTestResult)->Void)) {
        let testResult = ModelTestResult()
        testResult.model = model
        
        let modelUrlStr = "\(hostUrlStr)/getFile/\(model.name)"
        var modelRequest = URLRequest(url: URL(string: "\(modelUrlStr)/model")!)
        modelRequest.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        Alamofire.request(modelRequest).validate().responseData { (response) in
            guard response.result.isSuccess, let modelData = response.result.value else {
                let msg = "get model \(model.name) error"
                self.testLog(msg)
                testResult.msg = msg
                completion(testResult)
                return
            }
            //let modelData2 = try! Data(contentsOf: URL(fileURLWithPath: Bundle.main.path(forResource: "yolo_model_v3_16", ofType: nil)!))
            let modelPtr = UnsafeMutablePointer<UInt8>.allocate(capacity: modelData.count)
            NSData(data: modelData).getBytes(modelPtr, length: modelData.count)
            var paramsRequest = URLRequest(url: URL(string: "\(modelUrlStr)/params/\(model.paramsPrecision)")!)
            paramsRequest.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
            Alamofire.request(paramsRequest).validate().responseData(completionHandler: { (response) in
                guard response.result.isSuccess, let paramsData = response.result.value else {
                    let msg = "get params \(model.name) error"
                    self.testLog(msg)
                    testResult.msg = msg
                    completion(testResult)
                    return
                }
                //let paramsData2 = try! Data(contentsOf: URL(fileURLWithPath: Bundle.main.path(forResource: "yolo_params_v3_16", ofType: nil)!))
                let paramsPtr = UnsafeMutablePointer<UInt8>.allocate(capacity: paramsData.count)
                //paramsPtr.copyMemory(from: NSData(data: paramsData2).bytes, byteCount: paramsData2.count)
                NSData(data: paramsData).getBytes(paramsPtr, length: paramsData.count)
                guard let net = try? Net(device: device, inParamPointer: paramsPtr, inParamSize: paramsData.count, inModelPointer: modelPtr, inModelSize: modelData.count) else {
                    let msg = "init net error"
                    self.testLog(msg)
                    testResult.msg = msg
                    completion(testResult)
                    return
                }
                net.inputDim = Dim(inDim: [1, 1, 1, 1])
                net.paramPrecision = (model.paramsPrecision == 0) ? .Float16 : .Float32
                net.metalLibPath = Bundle.main.path(forResource: "paddle-mobile-metallib", ofType: "metallib")
                net.metalLoadMode = .LoadMetalInCustomMetalLib
                guard let runner = try? Runner(inNet: net, commandQueue: commandQueue), runner.load() else {
                    let msg = "init runner error"
                    self.testLog(msg)
                    testResult.msg = msg
                    completion(testResult)
                    return
                }
                guard let feedVar = runner.feedOpOutputVarDesc(), let dims = feedVar.dims, let fetchVars = runner.fetchOpInputVarDesc(), fetchVars.count > 0 else {
                    let msg = "feed var nil"
                    self.testLog(msg)
                    testResult.msg = msg
                    completion(testResult)
                    return
                }
                let fetchVar = fetchVars[0]
                net.inputDim = Dim(inDim: [dims[0], dims[2], dims[3], dims[1]])
               
                var feedVarRequest = URLRequest(url: URL(string: "\(modelUrlStr)/data/\(feedVar.name.replacingOccurrences(of: "/", with: "_"))")!)
                feedVarRequest.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
                Alamofire.request(feedVarRequest).validate().responseData(completionHandler: { (response) in
                    guard response.result.isSuccess, let inputData = response.result.value else {
                        let msg = "get var \(feedVar) error"
                        self.testLog(msg)
                        testResult.msg = msg
                        completion(testResult)
                        return
                    }
                    var fetchvarRequest = URLRequest(url: URL(string: "\(modelUrlStr)/data/\(fetchVar.name.replacingOccurrences(of: "/", with: "_"))")!)
                    fetchvarRequest.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
                    Alamofire.request(fetchvarRequest).validate().responseData(completionHandler: { (response) in
                        guard response.result.isSuccess, let outputData = response.result.value else {
                            let msg = "get var \(fetchVar) error"
                            self.testLog(msg)
                            testResult.msg = msg
                            completion(testResult)
                            return
                        }
                        self.runModel(model: model, net: net, inputData: inputData, outputData: outputData, compareCompletion: { (testResult) in
                            paramsPtr.deallocate()
                            modelPtr.deallocate()
                            completion(testResult)
                        })
                    })
                })
            })
        }
    }
    
    private func runModel(model: TestModel, net: Net, inputData: Data, outputData: Data, compareCompletion: @escaping ((_ testResult: ModelTestResult)->Void)) {
        let testResult = ModelTestResult()
        testResult.model = model
        
        net.useMPS = model.useMPS
        net.paramPrecision = (model.paramsPrecision == 0) ? .Float16 : .Float32
        guard let runner = try? Runner(inNet: net, commandQueue: commandQueue), runner.load(optimizeProgram: model.fusion, optimizeMemory: model.reuseTexture) else {
            let msg = "runner init or load fail"
            self.testLog(msg)
            testResult.msg = msg
            compareCompletion(testResult)
            return
        }
        self.runner = runner
        let buffer = device.makeBuffer(length: inputData.count, options: [])!
        memcpy(buffer.contents(), NSData(data: inputData).bytes, inputData.count)
        runner.getTexture(inBuffer: buffer, getTexture: { (success, texture) in
            if success, let texture = texture {
                runner.predict(texture: texture, completion: { (success, results) in
                    if success, let resultHolders = results {
                        let resultHolder = resultHolders[0]
                        let outputSize = outputData.count/MemoryLayout<Float32>.stride
                        let output = NSData(data: outputData).bytes.bindMemory(to: Float32.self, capacity: outputSize)
                        guard resultHolder.capacity == outputSize else {
                            let msg = "count not equal"
                            self.testLog(msg)
                            testResult.msg = msg
                            compareCompletion(testResult)
                            return
                        }
                        let precision = model.diffPrecision
                        var isResultEqual = true
                        var msg = ""
                        for i in 0..<outputSize {
                            let a = output[i]
                            let b = resultHolder.result[i]
                            // && abs(a - b) / min(abs(a), abs(b)) > 0.05 
                            if abs(a - b) > precision {
                                isResultEqual = false
                                msg = "unequal: i: \(i) target: \(output[i]) result: \(resultHolder.result[i])"
                                self.testLog(msg)
                                testResult.msg = msg
                                break
                            }
                        }
                        testResult.isResultEqual = isResultEqual
                        func checkPerformance(testResult: ModelTestResult, completion: @escaping ((_ testResult: ModelTestResult)->Void)) {
                            if model.testPerformance {
                                DispatchQueue.main.asyncAfter(deadline: .now() + 0.2, execute: {
                                    self.testPerformance(runner: runner, texture: texture, testResult: testResult, completion: completion)
                                })
                            } else {
                                completion(testResult)
                            }
                        }
                        if isResultEqual {
                            checkPerformance(testResult: testResult, completion: compareCompletion)
                        } else {
                            if model.reuseTexture {
                                testResult.diverVarName = "can not find diver var when reusing texture"
                                checkPerformance(testResult: testResult, completion: compareCompletion)
                            } else {
                                self.findDivergentVar(runner: runner, testResult: testResult, completion: { (testResult) in
                                    checkPerformance(testResult: testResult, completion: compareCompletion)
                                })
                            }
                        }
                        return
                    }
                })
            } else {
                let msg = "get texture error"
                self.testLog(msg)
                testResult.msg = msg
                compareCompletion(testResult)
            }
        }, channelNum: net.inputDim[3])
    }
    
    private func testPerformance(runner: Runner, texture: MTLTexture, testResult: ModelTestResult, completion: @escaping ((ModelTestResult)->Void)) {
        let startTime = CFAbsoluteTimeGetCurrent()
        runner.predict(texture: texture) { (success, results) in
            self.testLog("!!!!!\(CFAbsoluteTimeGetCurrent()-startTime)")
            if success && count > 0 {
                timeCosts.append(CFAbsoluteTimeGetCurrent()-startTime)
            }
            count += 1
            if count < totalCount {
                DispatchQueue.main.asyncAfter(deadline: .now()+0.1, execute: {
                    self.testPerformance(runner: runner, texture: texture, testResult: testResult, completion: completion)
                })
            } else {
                if timeCosts.count > 0 {
                    var total = 0.0
                    for time in timeCosts {
                        total += time
                    }
                    let avgTime = total / Double(timeCosts.count)
                    testResult.performance = avgTime
                }
                count = 0
                timeCosts.removeAll()
                completion(testResult)
            }
        }
    }
    
    private func findDivergentVar(runner: Runner, testResult: ModelTestResult, completion: @escaping ((ModelTestResult)->Void)) {
        orderedVars.removeAll()
        varIndex = 0
        orderedVars = runner.getAllOutputVars()
        compareVars(runner: runner, model: testResult.model!) { (varName) in
            testResult.diverVarName = varName
            completion(testResult)
        }
    }
    
    private func compareVars(runner: Runner, model: TestModel, completion: @escaping ((_ varName: String)->Void)) {
        let severVars: [String] = Array(model.varsDic.keys)
        let urlString: String = "\(hostUrlStr)/getFile/\(model.name)/data"
        if varIndex < orderedVars.count {
            var varName = orderedVars[varIndex]
            while !(severVars.contains(varName)) || runner.fetchVar(varName) == nil {
                varIndex += 1
                if varIndex < orderedVars.count {
                    varName = orderedVars[varIndex]
                } else {
                    break
                }
            }
            if severVars.contains(varName) {
                var severVarRequest = URLRequest(url: URL(string: "\(urlString)/\(varName)")!)
                severVarRequest.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
                Alamofire.request(severVarRequest).validate().responseData { (response) in
                    varIndex += 1
                    guard response.result.isSuccess, let varData = response.result.value else {
                        self.compareVars(runner: runner, model: model, completion: completion)
                        return
                    }
                    let varSize = varData.count/MemoryLayout<Float32>.stride
                    let varTarget = NSData(data: varData).bytes.bindMemory(to: Float32.self, capacity: varSize)
                    let varResult = runner.fetchVar(varName)!
                    guard varSize == varResult.count else {
                        completion(varName)
                        return
                    }
                    let precision = model.diffPrecision
                    for i in 0..<varSize {
                        if abs(varTarget[i] - varResult[i]) > precision {
                            completion(varName)
                            return
                        }
                    }
                    self.compareVars(runner: runner, model: model, completion: completion)
                    return
                }
            } else {
                completion("")
            }
        }
    }
    
    private func testModelsFromModel(_ model: Model) -> [TestModel] {
        var testModels: [TestModel] = []
        for paramsPrecision in model.paramsPrecision {
            for fusion in model.fusion {
                for useMPS in model.useMPS {
                    for reuseTexture in model.reuseTexture {
                        let testModel = TestModel(name: model.name, paramsPrecision: paramsPrecision, fusion: fusion, useMPS: useMPS, reuseTexture: reuseTexture, testPerformance: model.testPerformance, diffPrecision: model.diffPrecision, varsDic: model.varsDic)
                        testModels.append(testModel)
                    }
                }
            }
        }
        return testModels
    }
    
    private func testLog(_ log: String) {
        DispatchQueue.main.async {
            self.textView.text = self.textView.text + "\n\(log)"
            DispatchQueue.main.asyncAfter(deadline: .now()+0.1) {
                self.scrollTextViewToBottom(textView: self.textView)
            }
        }
    }
    
    private func scrollTextViewToBottom(textView: UITextView) {
        if textView.text.count > 0 {
            let location = textView.text.count - 1
            let bottom = NSMakeRange(location, 1)
            textView.scrollRangeToVisible(bottom)
        }
    }
}
