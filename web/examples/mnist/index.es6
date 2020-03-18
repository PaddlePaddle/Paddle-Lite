import 'babel-polyfill';
import Paddle from '../../src/paddle/paddle';
import IO from '../../src/feed/imageFeed';
/**
 * @file model demo mnist 入口文件
 * @author wangqun@baidu.com
 *
 */

const pic = document.getElementById('pic');
const io = new IO();
let model = {};
async function run() {

    let feed = io.process({
        input: pic,
        params: {
            targetShape: [1, 3, 320, 320], // 目标形状 为了兼容之前的逻辑所以改个名
            scale: 256, // 缩放尺寸
            width: 224, height: 224, // 压缩宽高
            shape: [3, 224, 224], // 预设tensor形状
            mean: [0.485, 0.456, 0.406], // 预设期望
            std: [0.229, 0.224, 0.225]  // 预设方差
        }});

    console.dir(['feed', feed]);

    const path = 'model/mnist';

    const MODEL_CONFIG = {
        dir: `/${path}/`, // 存放模型的文件夹
        main: 'model.json', // 主文件
    };
    const paddle = new Paddle({
        urlConf: MODEL_CONFIG,
        options: {
            multipart: false,
            dataType: 'json'
        }
    });

    model = await paddle.load();



    let inst = model.execute({
        input: feed
    });

    // 其实这里应该有个fetch的执行调用或者fetch的输出
    let result = await inst.read();

    // let inst = model.execute({input: cat});
    // let res = inst.read();
    console.dir(['result', result]);
    // var fileDownload = require('js-file-download');
    // fileDownload(res, 'result.csv');
}
run();