import 'babel-polyfill';
// import model from '../data/model.test2';
// import model from '../data/model.test.conv2d';
import GraphExecutor from '../../src/executor/executor';
import Loader from '../../src/executor/loader';
import Runtime from '../../src/runtime/runtime';
// 获取map表
import Map from '../data/map';
console.dir(['map', Map]);

let Diff = require('./diff');
let datas;
let otherResult;
let output
async function run() {
    const MODEL_URL = '/test/unitData/model.test.batchnorm.json';
    const graphModel= new Loader();
    const model = await graphModel.loadGraphModel(MODEL_URL);
    datas = model.handler;
    output = deepCopy(model.handler);
    // 测试单元
    let item = getTensor('batchnorm');
    func(item);
    // let inst = model.execute({input: cat});
    // console.dir(['result', inst.read()]);
}
run();

function deepCopy (data) {
    return JSON.parse(JSON.stringify(data));
}

// let output = deepCopy(datas);
let getTensor = function(id, times = 1) {
    let find = 0;
    let data = datas.ops.filter((item, idx) => {
        if (id === item.type) {
            ++find;
            if (find === times) {
                return true;
            }
        }
    });
    return getInputs(data[0]);
};

let getInputs = function(data) {

    Object.keys(data.inputs).forEach(function(key){
        data.inputs[key] = getValue(data.inputs[key][0], datas);

    });
    Object.keys(data.outputs).forEach(function(key){
        let out = getValue(data.outputs[key][0], datas)
        data.outputs[key] = out;
        otherResult = out[0].data;
    });
    return data;

};

let getResult = function(id) {
    let data = output.ops.filter((item, idx) => {
        if (id === item.type) {

            return true;
        }
    });
    return getoutputs(data[0]);
};
let getoutputs = function(data) {
    let otherResult;
    Object.keys(data.outputs).forEach(function(key){
        let out = getValue(data.outputs[key][0], output);
        otherResult = out[0].data;
    });
    return otherResult;
};

let getValue = function(name, datas) {
    return datas.vars.filter((item, idx) => {
        if (name === item.name) {
            return item;
        }
    });
};
// // 测试单元
// let item = getTensor('conv2d');

let func = function (item) {
    let inst = Runtime.init({
        'width_raw_canvas': 512,
        'height_raw_canvas': 512
    });
    const executor = new GraphExecutor(item);
    executor.execute(executor, {}, inst);
    console.dir(['result', inst.read()]);


    // var one = inst.read();
    // var other = getResult('softmax');
    // var color ='';
    // var span = null;

    // var diff = Diff.diffChars(one.toString(), other.toString()),
    //     display = document.getElementById('display'),
    //     fragment = document.createDocumentFragment();
    //
    // diff.forEach(function(part){
    //     // green for additions, red for deletions
    //     // grey for common parts
    //     color = part.added ? 'green' :
    //         part.removed ? 'red' : 'grey';
    //     span = document.createElement('span');
    //     span.style.color = color;
    //     span.appendChild(document
    //         .createTextNode(part.value));
    //     fragment.appendChild(span);
    // });
    //
    // display.appendChild(fragment);

};

