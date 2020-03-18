// import VConsole from 'vconsole';
import 'babel-polyfill';
import Paddle from '../../src/paddle/paddle';
import IO from '../../src/feed/imageFeed';
// import Logger from '../../tools/logger';
// window.log = new Logger();
// // 统计参数
// window.badCases = [];

// 后处理测试用例
// let tempPic = [demoPic, demoPic2, demoPic3, demoPic4, demoPic5];
/**
 * @file model demo 入口文件
 * @author wangqun@baidu.com
 *
 */
// 模型输出shape
const outputShapes = {
    '608': {
        from: [19, 19, 25, 1],
        to: [19, 19, 5, 5]
    },
    '320': {
        from: [10, 10, 25, 1],
        to: [10, 10, 5, 5]
    },
    '320fused': {
        from: [10, 10, 25, 1],
        to: [10, 10, 5, 5]
    },
    'tinyYolo': {
        from: [10, 10, 25, 1],
        to: [10, 10, 5, 5]
    }
};
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
    'tinyYolo': {
        fw: 320,
        fh: 320
    }
};
// 模型路径
const modelPath = {
    'tinyYolo': 'model/tinyYolo'
};
const modelType = 'tinyYolo';
const path = modelPath[modelType];
// 统计参数
let loaded = false;
let model = {};
window.statistic = [];
const {fw, fh} = feedShape[modelType];
// 第一遍执行比较慢 所以预热一下
async function run(input) {
    // const input = document.getElementById('mobilenet');
    //log.start('总耗时');
    const io = new IO();
    // log.start('预处理');
    let feed = io.process({
        input: input,
        params: {
            gapFillWith: '#000', // 缩放后用什么填充不足方形部分
            targetSize: {
                height: fw,
                width: fh
            },
            targetShape: [1, 3, fh, fw], // 目标形状 为了兼容之前的逻辑所以改个名
            // shape: [3, 608, 608], // 预设tensor形状
            mean: [117.001, 114.697, 97.404], // 预设期望
            // std: [0.229, 0.224, 0.225]  // 预设方差
        }
    });
    // log.end('预处理');
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
                dataType: 'binary',
                options: {
                    fileCount: 1, // 切成了多少文件
                    getFileName(i) { // 获取第i个文件的名称
                        return 'chunk_0.dat';
                    }
                }
            }
        });

        model = await paddle.load();

    }

    let inst = model.execute({
        input: feed
    });

    // 其实这里应该有个fetch的执行调用或者fetch的输出
    let result = await inst.read();
    // log.end('运行耗时');
    // log.end('后处理-读取数据');
    console.dir(['result', result]);
    //log.start('后处理-形状调整');
    const newData = [];
    let newIndex = -1;
    const [w, h, c, b] = outputShapes[modelType].from;
    // c channel
    for (let i = 0; i < c; i++) {
        // height channel
        for (let j = 0; j < h; j++) {
            // width channel
            for (let k = 0; k < w; k++) {
                // position: (0, 0, 0, 0)
                const index = j * (c * h) + k * c + i;
                // const index = j * (i * k) + k * i + i;
                newData[++newIndex] = result[index];
            }
        }
    }
    // log.end('后处理-形状调整');
    // log.start('后处理-画框');
    testRun(newData, input);
    // log.end('后处理-画框');
    // log.end('后处理');
    // log.end('总耗时');
}
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
            //log.during('每次执行的时间间隔');
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

/* 后处理图片 by zhangmiao06 */
let preTestRun = (index) => {
    let img = document.getElementById('image');
    img.src = tempPic[index];
    img.onload = function() {
        testRun(testOutput.data[index], img);
    };
};
let testRun = (data, img) => {
    // console.log('ori', data);
    const {from, to} = outputShapes[modelType];
    // let shape = [1, 25, 19, 19];
    let shape = [].concat(from).reverse();
    // 1.从一维数组到1*25*19*19
    let formatData = reshapeMany({
        data: data,
        reshapeShape: shape
    });
    // console.log('一维到多维', formatData);
    // 2.从1*25*19*19 到 19*19*25*1
    let formatData2 = transpose({
        data: formatData,
        shape: shape,
        transposeShape: [2, 3, 1, 0]
    });
    // console.log('transpose', formatData2);
    // 3.从19*19*25*1到19*19*5*5
    let formatData3 = reshape({
        data: formatData2,
        shape: from,
        reshapeShape: to
    });
    // console.log('reshape', formatData3);
    // 4.运算
    let finalData = handleFinal(formatData3, shape, img);
    // console.log('final', finalData);
    // 5.处理画布
    // handleCanvas(finalData, img);
    handleDiv(finalData, img);
};

// sigmoid
let sigmoid = (x) => {
    if (x < -100) {
        return 0.0;
    }
    return 1 / (1 + Math.exp(-x));
}

// transpose
let transpose = (data) => {
    let shape = data.shape;
    let transposeShape = data.transposeShape;
    let formatData = data.data;
    let formatData2 = [];
    for(let n = 0; n < shape[transposeShape[0]]; n++) {
        let nData = [];
        for(let c = 0; c < shape[transposeShape[1]]; c++) {
            let cData = [];
            for(let row = 0; row < shape[transposeShape[2]]; row++) {
                let rowData = [];
                for(let col = 0; col < shape[transposeShape[3]]; col++) {
                    let tempArr = [n, c, row, col];
                    let newN = n;
                    let newC = c;
                    let newW = row;
                    let newH = col;
                    transposeShape.forEach((item, index)=> {
                        switch(item) {
                            case 0:
                                newN = tempArr[index];
                                break;
                            case 1:
                                newC = tempArr[index];
                                break;
                            case 2:
                                newW = tempArr[index];
                                break;
                            case 3:
                                newH = tempArr[index];
                        }
                    });
                    rowData.push(formatData[newN][newC][newW][newH]);
                }
                cData.push(rowData);
            }
            nData.push(cData);
        }
        formatData2.push(nData);
    }
    return formatData2;
};

// reshape
let reshape = (data) =>{
    let formatData2 = data.data;
    let shape = data.shape;
    let reshapeShape = data.reshapeShape;
    // 1.变成一维
    let tempData = reshapeOne({
        data: formatData2,
        shape: shape
    });
    // 2.变成多维
    let formatData3 = reshapeMany({
        data: tempData,
        reshapeShape: reshapeShape
    });
    return formatData3;
};

// 变成一维
let reshapeOne = (data) => {
    let formatData2 = data.data;
    let shape = data.shape;
    let tempData = [];
    for(let n = 0; n < shape[0]; n++) {
        for(let c = 0; c < shape[1]; c++) {
            for(let row = 0; row < shape[2]; row++) {
                for(let col = 0; col < shape[3]; col++) {
                    tempData.push(formatData2[n][c][row][col]);
                }
            }
        }
    }
    return tempData;
};

// 变成多维
let reshapeMany = (data) => {
    let tempData = data.data;
    let reshapeShape = data.reshapeShape;
    let formatData3 = [];
    for(let n = 0; n < reshapeShape[0]; n++) {
        let nData = [];
        for(let c = 0; c < reshapeShape[1]; c++) {
            let cData = [];
            for(let row = 0; row < reshapeShape[2]; row++) {
                let rowData = [];
                for(let col = 0; col < reshapeShape[3]; col++) {
                    let tempN = n * reshapeShape[1] * reshapeShape[2] * reshapeShape[3];
                    let tempC = c * reshapeShape[2] * reshapeShape[3];
                    let tempRow = row * reshapeShape[3];
                    rowData.push(tempData[tempN + tempC + tempRow + col]);
                }
                cData.push(rowData);
            }
            nData.push(cData);
        }
        formatData3.push(nData);
    }
    return formatData3;
};
let calSize = (img) => {
    let w1 = img.width;
    let h1 = img.height;
    let wh1 = Math.max(w1, h1);
    // let factor = 608.0 / wh1;
    let factor = fw / wh1;
    let width = Math.round(w1 * factor);
    let height = Math.round(h1 * factor);
    return [w1, h1, width, height];
};
// 处理运算
let handleFinal = (formatData3, shape, img) => {
    let finalData = [];
    let c = shape[2];
    let [w1, h1, width, height] = calSize(img);
    let factorX = Math.max(width, height) / width;
    let factorY = Math.max(width, height) / height;

    let maxProb = 0.0;
    let anchors = [[1.603231, 2.094468], [6.041143, 7.080126], [2.882459, 3.518061], [4.266906, 5.178857], [9.041765, 10.66308]];

    for(let i = 0; i < shape[2]; i++) {
        for(let j = 0; j < shape[3]; j++) {
            for(let k = 0; k < anchors.length; k++) {
                let [a1, a2, a3, a4, prob] = formatData3[i][j][k];
                prob = sigmoid(prob);
                if (prob > maxProb && prob >= 0.5) {
                    let ctx = (j + sigmoid(a1)) / c * factorX;
                    let cty = (i + sigmoid(a2)) / c * factorY;
                    let col = Math.exp(a3) * anchors[k][0] / c * factorX;
                    let row = Math.exp(a4) * anchors[k][1] / c * factorY;
                    let x = (ctx - (col / 2));
                    let y = (cty - (row / 2));
                    finalData.push([x * w1, y * h1, col * w1, row * h1, prob]);
                }
            }
        }
    }
    return finalData;
};

// 处理画布
let handleCanvas = (finalData, img) => {
    let myCanvas = document.getElementById('myCanvas');
    let [w1, h1, width, height] = calSize(img);
    myCanvas.width = w1;
    myCanvas.height = h1;
    let ctx = myCanvas.getContext("2d");
    ctx.drawImage(img, 0, 0, w1, h1);

    finalData.forEach((demoArr,index) => {
        let [demoLeft, demoTop, demoWidth, demoHeight, prob] = demoArr;
        ctx.beginPath();
        ctx.strokeStyle="red";
        ctx.moveTo(demoLeft, demoTop);
        ctx.lineTo(demoLeft + demoWidth, demoTop);
        ctx.lineTo(demoLeft + demoWidth, demoTop + demoHeight);
        ctx.lineTo(demoLeft, demoTop + demoHeight);
        ctx.closePath();
        ctx.stroke();
    });
};
let handleDiv = (finalData, img) => {
    if (finalData.length < 1) {
        return false;
    }
    let myCanvas = document.getElementById('myDiv');
    let maxIndex = 0;
    if (finalData.length > 1) {
        for(let i = 1; i < finalData.length; i++) {
            if (finalData[i].prob > finalData[maxIndex].prob) {
                maxIndex = i;
            }
        }
    }
    let [demoLeft, demoTop, demoWidth, demoHeight, prob] = finalData[maxIndex];
    myCanvas.style.width = demoWidth;
    myCanvas.style.height = demoHeight;
    myCanvas.style.left = demoLeft;
    myCanvas.style.top = demoTop;
};
// preTestRun(0);

// run(document.getElementById('pic'));
