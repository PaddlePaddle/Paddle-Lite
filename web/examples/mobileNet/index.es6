import 'babel-polyfill';
import Paddle from '../../src/paddle/paddle';
import IO from '../../src/feed/imageFeed';
import Utils from '../../src/utils/utils';
// 获取map表
import Map from '../../test/data/map';
/**
 * @file model demo 入口文件
 * @author wangqun@baidu.com
 *
 */
// 模型feed数据
const feedShape = {
    '608': {
        fw: 608,
        fh: 608
    },
    '320': {
        fw: 320,
        fh: 320
    },
    '320fused': {
        fw: 320,
        fh: 320
    },
    'separate': {
        fw: 244,
        fh: 244
    }
};
const modelType = 'separate';
const {fw, fh} = feedShape[modelType];
// 统计参数
let loaded = false;
let model = {};
window.statistic = [];
async function run(input) {
    // const input = document.getElementById('mobilenet');
    const io = new IO();

    let feed = io.process({
        input: input,
        params: {
            targetShape: [1, 3, fh, fw], // 目标形状 为了兼容之前的逻辑所以改个名
            scale: 256, // 缩放尺寸
            width: 224, height: 224, // 压缩宽高
            shape: [3, 224, 224], // 预设tensor形状
            mean: [0.485, 0.456, 0.406], // 预设期望
            std: [0.229, 0.224, 0.225]  // 预设方差
        }});

    console.log('feed', feed);
    const path = 'model/mobileNet';

    if (!loaded) {
        const MODEL_CONFIG = {
            dir: `/${path}/`, // 存放模型的文件夹
            main: 'model.json', // 主文件
        };
        loaded = true;
        const paddle = new Paddle({
            urlConf: MODEL_CONFIG,
            options: {
                multipart: true,
                dataType: 'json'
            }
        });

        model = await paddle.load();

    }

    let inst = model.execute({
        input: feed
    });

    // 其实这里应该有个fetch的执行调用或者fetch的输出
    let result = await inst.read();

    console.dir(['result', result]);
   // let maxItem = Utils.getMaxItem(result);
  //  document.getElementById('txt').innerHTML = Map['' + maxItem.index];
   // console.log('识别出的结果是' + Map['' + maxItem.index]);
    // console.dir(['每个op耗时', window.statistic]);
    // let total = statistic.reduce((all, cur) => {
    //     return all + cur.runTime;
    // }, 0);
    // console.log('op total = ' + total);

};
var image = '';
function selectImage(file) {
    if (!file.files || !file.files[0]) {
        return;
    }
    let reader = new FileReader();
    reader.onload = function (evt) {
        let img = document.getElementById('image');
        img.src = evt.target.result;
        img.onload = function() {
            run(img);
        };
        image = evt.target.result;
    }
    reader.readAsDataURL(file.files[0]);
}
// selectImage
document.getElementById("uploadImg").onchange = function () {
    selectImage(this);
};
