/* eslint-disable */
import Utils from './utils';
import Tensor from './tensor';
/**
 * @file op的数据对象
 * @author yangmingming
 *
 */
const keys = [
    'paddings',
    'strides',
    'dilations',
    'ksize'
];
// 从tensor对象中获取的数据
const tensorAttrs = [
    'length_shape',
    'width_shape',
    'height_shape',
    'width_texture',
    'height_texture',
    'channel',
    'total_shape'
];
// shader中需要的常量
const shaderAttrs = {
    scale: {
        'bias': 'bias_value',
        'scale': 'multi_value'
    },
    pool2d: {
        'pooling_type': 'type_pool'
    }
};
// model的名字和paddle web的tensor名字mapping
const tensorName = {
    'input': 'origin',
    'x': 'origin',
    'filter': 'filter',
    'y': 'counter',
    'output': 'out',
    'out': 'out',
    'scale': 'scale',
    'bias': 'bias',
    'mean': 'mean',
    'variance': 'variance'
};
// unique behavior
const opBehavior = {
    conv2d: [
        'needBatch'
    ],
    batchnorm: [
        'needBatch',
        'mergeTensor'
    ],
    elementwise_add: [
        'broadcast',
        'needBatch'
    ],
    pool2d: [
        'isMax',
        'needBatch',
        'isGlobalPooling'
    ],
    relu: [
        'transToPrelu',
        'needBatch'
    ],
    leaky_relu: [
        'transToLeakyrelu',
        'needBatch'
    ],
    mul: [
        'reshape',
        'needBatch'
    ]
};
export default class OpData {
    constructor(name, input = {}, output = {}, attrs = {}) {
        this.name = name;
        this.attrs = attrs;
        // 是否忽略当前当前op, 使用dropout
        this.isPass = this.checkIsPass();
        if (this.isPass) {
            this.input = input;
            this.output = output;
            // op数据, 当前不扩展
            this.data = {
                'active_function': 'scale',
                'multi_value': '1.0',
                'bias_value': '0.0'
            };
            // tensor数据
            this.tensor = {};
            this.buildTensor();
            this.buildAttrs();
        }
    }

    buildTensor() {
        // todo: 是否需要形状对齐
        // todo: 是否需要广播tensor
        const tensorData = [];
        for (let key in this.input) {
            if (this.input.hasOwnProperty(key)) {
                const data = this.input[key] || [{}];
                // 默认取第一个数据
                if (tensorName[key.toLowerCase()]) {
                    data[0].tensorName = tensorName[key.toLowerCase()];
                    tensorData.push(data[0]);
                }
            }
        }
        // debugger
        // todo: 临时删除output里的Y
        delete this.output.Y;
        // 输出tensor
        for (let key in this.output) {
            if (this.output.hasOwnProperty(key)) {
                // 默认取第一个数据
                const data = this.output[key] || [{}];
                if (tensorName[key.toLowerCase()]) {
                    data[0].tensorName = tensorName[key.toLowerCase()];
                    tensorData.push(data[0]);
                }
            }
        }
        // unique behavior
        const behavior = opBehavior[this.name] || [];
        behavior.forEach(behavior => {
            this[behavior](tensorData);
        });
        // 生成tensor对象
        tensorData.forEach(data => {
            if (data) {
                if (data.notTensor) {
                    this.tensor[data.tensorName] = {
                        name: data.tensorName,
                        data: new Float32Array(data.data),
                        total_shape: data.data.length
                    };
                } else {
                    this.tensor[data.tensorName] = new Tensor({
                        name: data.tensorName,
                        shape: data.shape,
                        data: data.data,
                        needBatch: data.needBatch || false,
                        notCompressed: data.notCompressed || false
                    });
                }
            }
        });
        // console.dir(['tensors', this.tensor]);
    }

    buildAttrs() {
        // 计算属性
        for (let key in this.attrs) {
            if (this.attrs.hasOwnProperty(key)) {
                const item = this.attrs[key];
                if (Object.prototype.toString.call(item) === '[object Array]') {
                    if (keys.indexOf(key) > -1) {
                        this.data[key + '_x'] = item[0];
                        this.data[key + '_y'] = item[1];
                    }
                } else {
                    this.data[key] = item;
                    // 获取shader所需的数据
                    let shaderAttr = shaderAttrs[this.name];
                    if (shaderAttr && shaderAttr.hasOwnProperty(key)) {
                        this.data[shaderAttr[key]] = item;
                    }
                }
            }
        }
        // 获取tensor的数据
        for (let key in this.tensor) {
            const tensor = this.tensor[key];
            tensorAttrs.forEach(attr => {
                this.data[attr+ '_' + tensor.name] = tensor[attr];
            });
        }
    }

    needBatch(tensorData = []) {
        tensorData.forEach(data => (data.needBatch = true));
    }

    isGlobalPooling(tensorData = []) {
        let counter = tensorData.filter(tensor => (tensor.tensorName === 'origin'))[0] || {};
        let length = counter.shape && counter.shape.length || 0;
        if (length > 2 && this.attrs['global_pooling']) {
            this.attrs.ksize = [counter.shape[length - 2], counter.shape[length - 1]];
        }
    }

    broadcast(tensorData = []) {
        const x = tensorData[0];
        const y = tensorData[1];
        let small = y;
        if (x.shape.length - y.shape.length < 0) {
            small = x;
        }
        // face model
        small.notTensor = true;
        return;

        // mobilenet model
        // todo: 默认y的shape length是1, 以后需要实现通用版本
        let shape = Utils.getBroadcastShapeInPaddle(x.shape, y.shape, this.attrs['axis']);
        // 填充shape数据
        if (small.shape.length === 1) {
            const result = [];
            small.shape = shape;
            let total = shape.reduce((all, num) => all * num);
            for (let i = 0; i < small.shape[0]; i++) {
                let item = small.data[i];
                for (let j = 0; j < total / shape[0]; j++) {
                    result.push(item);
                }
            }
            small.data = result;
        }
    }

    isMax(tensorData = []) {
        const type = this.attrs['pooling_type'] === 'max' ? 1 : 0;
        this.attrs['pooling_type'] = type;
    }

    transToPrelu(tensorData = []) {
        this.data['multi_value'] = '0.0';
        this.data['active_function'] = 'prelu';
    }

    transToLeakyrelu(tensorData = []) {
        this.data['multi_value'] = this.attrs.alpha;
        this.data['active_function'] = 'leakyRelu';
        this.name = 'relu';
    }

    setActiveFunc(tensorData = []) {
        this.data['multi_value'] = '0.0';
        this.data['active_function'] = 'softmax';

    }

    reshape(tensorData = []) {
        let input = tensorData[0];
        let counter = tensorData[1];
        if (counter.shape.length > input.shape.length) {
            input = tensorData[1];
            counter = tensorData[0];
        }
        if (input.shape.length > 2 && counter.shape.length === 2) {
            let shape = Utils.getReshapeInPaddle(input.shape, counter.shape, tensorData[2].shape);
            input.shape = shape;
        }

    }

    mergeTensor(tensorData = []) {
        // 融合scale、bias、variance、mean
        let constants = ['scale', 'bias', 'variance', 'mean'];
        let result = {};
        let data = [];
        tensorData.forEach((tensor, index) => {
            result[tensor.tensorName] = tensor;
            result[tensor.tensorName + 'Index'] = index;
        });
        for (let i = 0; i < result[constants[0]].shape[0]; i++) {
            data.push(result[constants[0]].data[i]);
            data.push(result[constants[1]].data[i]);
            data.push(result[constants[2]].data[i]);
            data.push(result[constants[3]].data[i]);
        }
        tensorData[result[constants[0] + 'Index']].data = data;
        // 充分利用shader空间
        tensorData[result[constants[0] + 'Index']].notCompressed = true;
        tensorData[result[constants[0] + 'Index']].shape[0] *= 4;
        tensorData.splice(result[constants[1] + 'Index'], 1, 0);
        tensorData.splice(result[constants[2] + 'Index'], 1, 0);
        tensorData.splice(result[constants[3] + 'Index'], 1, 0);
    }

    checkIsPass() {
        if (this.name === 'dropout') {
            if (this.attrs['dropout_implementation'] === 'downgrade_in_infer') {
                this.name = 'scale';
                this.attrs['scale'] = this.attrs['dropout_prob'];
                this.attrs['bias'] = 0.0;
                return true;
            }
            return false;
        }
        if (this.name === 'depthwise_conv2d') {
            this.name = 'conv2d';
        }
        return true;
    }

    dispose() {
        this.input = null;
        this.output = null;
        this.attrs = null;
        for (let key in this.tensor) {
            this.tensor[key].dispose();
        }
        this.tensor = {};
    }
}
