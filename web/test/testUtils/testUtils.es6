import 'babel-polyfill';
import Paddle from '../../src/paddle/paddle';

const unitPath = {
    'conv2d': 'model.test.conv2d.json',
    'batchnorm': 'model.test.batchnorm.json',
    'mul': 'model.test.mul.json',
    'pool2d': 'model.test.pool2d.json',
    'relu': 'model.test.relu.json',
    'scale': 'model.test.scale.json',
    'softmax': 'model.test.softmax.json',
    'relu6' : 'model.test.relu6.json'
};
// 制定运行的 op
const modelType = 'softmax';
const unitData = unitPath[modelType];

let Diff = require('./diff');
let datas;
let otherResult;
let output
async function run() {
    const path = 'test/unitData';
    const MODEL_CONFIG = {
        dir: `/${path}/`, // 存放模型的文件夹
        main: unitData, // 主文件
    };

    const paddle = new Paddle({
        urlConf: MODEL_CONFIG,
        options: {
            test: true
        }
    });

    let model = await paddle.load();
    datas = model.graph.data;
    output = deepCopy(datas);
    // 测试单元
    model.graph.weightMap.forEach(op => {
        const type = op.type;
        if (type !== 'feed' && type !== 'fetch') {
            console.log(op.type);
            model.graph.buildOpData(op);
        }
    });
    const executor = model.graph.weightMap;
    let inst = model.graph.execute_(executor[0]);

    let result = model.graph.inst.read();
    console.dir(['result', result]);
    var one = model.graph.inst.read();
// var other = getResult('conv2d');

    console.log('one');
    console.log(one);
    console.log('other');
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

let func = function (model) {

  //  console.log(other);


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

