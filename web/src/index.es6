import 'babel-polyfill';
import Graph from './executor/loader';
import IO from './executor/io';
/**
 * @file model demo 入口文件
 * @author yangmingming@baidu.com
 *
 */
// 'http://mms-xr.cdn.bcebos.com/paddle/mnist/model.json'
const MODEL_URL = '../demo/model/model.json';
const graphModel = new Graph();
const model = graphModel.loadGraphModel(MODEL_URL);
const cat = document.getElementById('pic');
const io = new IO();

let inst = model.execute({input: cat});
let res = inst.read();
console.dir(['result', res]);
var fileDownload = require('js-file-download');
fileDownload(res, "result.csv");
