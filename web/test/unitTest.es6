import 'babel-polyfill';
import units from './units/units';
let qs = require('qs');
/**
 * @file 入口文件
 * @author wangqun@baidu.com
 *
 */
// 引入 op
const FSHADER_CON2D = require('../src/shader/f_elementwise_conv2d3_shader.c');


const shapeA = [1, 3, 256, 256];
const shapeB = [3];
const imgUrl = require('./data/banana.jpeg');
let shapeAData;
let shapeBData;
let inst;

const matrix = units.mockOrigin();
const filter = units.mockFilter();
// 原始张量，上下左右1个单位的padding，步长是1
let conf = {
    'filter_size_width': 3,
    'filter_size_height': 3,
    'origin_size_width': matrix.sx,
    'origin_size_height': matrix.sx,
    'out_size_width': 3,
    'out_size_height': 3,
    'stride_horizontal': 1,
    'stride_vertical': 1,
    'pad_left': 1,
    'pad_top': 1,
    'dilation_horizontal': 2,
    'dilation_vertical': 2
}
units.init(conf, FSHADER_CON2D).then(instance => {
    if (!instance || typeof instance === 'string') {
        throw new Error(instance || '不支持float texture');
    }
    inst = instance;
}).then(() => {
    console.dir(['卷积核', filter]);
    console.dir(['origin data', matrix.data]);
    // 执行conv2d
    inst.compute(filter, matrix.data, 'conv2d');
}).then(() => {
    // 读取结果
    const result = inst.read();
    console.dir(['conv2d的执行结果', result]);

    let input = {
        filter: filter,
        origin: matrix.data,
    };
    Object.assign(input, conf);
    console.dir(['完整input', input]);
    // console.dir(['完整输入和输出', params]);
    inst.getResult('pool2d', input, result);
}).catch(err => {
    console.log('-----------error---------' + err);
});
