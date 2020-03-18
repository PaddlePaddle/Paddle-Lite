/* eslint-disable */

/* 后处理图片 by zhangmiao06 */
// let preTestRun = index => {
//     let img = document.getElementById('image');
//     img.src = tempPic[index];
//     img.onload = function () {
//         testRun(testOutput.data[index], img);
//     };
// };

import models from '../utils/models';

const isSimilar = (r1, r2, threshold = 5) => {
    return Math.max(Math.abs(r1[0] - r2[0]), Math.abs(r1[1] - r2[1])) < threshold;
    // return Math.abs((r1[0] + r1[1] + r1[2] + r1[3]) - (r2[0] + r2[1] + r2[2] + r2[3])) < threshold;
}

// sigmoid
let sigmoid = (x) => {
    if (x < -100) {
        return 0.0;
    }

    return 1 / (1 + Math.exp(-x));
};

// transpose
let transpose = (data) => {
    let shape = data.shape;
    let transposeShape = data.transposeShape;
    let formatData = data.data;
    let formatData2 = [];
    for (let n = 0; n < shape[transposeShape[0]]; n++) {
        let nData = [];
        for (let c = 0; c < shape[transposeShape[1]]; c++) {
            let cData = [];
            for (let row = 0; row < shape[transposeShape[2]]; row++) {
                let rowData = [];
                for (let col = 0; col < shape[transposeShape[3]]; col++) {
                    let tempArr = [n, c, row, col];
                    let newN = n;
                    let newC = c;
                    let newW = row;
                    let newH = col;
                    transposeShape.forEach((item, index) => {
                        switch (item) {
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
const reshape = (data) => {
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
const reshapeOne = (data) => {
    let formatData2 = data.data;
    let shape = data.shape;
    let tempData = [];
    for (let n = 0; n < shape[0]; n++) {
        for (let c = 0; c < shape[1]; c++) {
            for (let row = 0; row < shape[2]; row++) {
                for (let col = 0; col < shape[3]; col++) {
                    tempData.push(formatData2[n][c][row][col]);
                }
            }
        }
    }
    return tempData;
};

// 变成多维
const reshapeMany = data => {
    let tempData = data.data;
    let reshapeShape = data.reshapeShape;
    let formatData3 = [];
    for (let n = 0; n < reshapeShape[0]; n++) {
        let nData = [];
        for (let c = 0; c < reshapeShape[1]; c++) {
            let cData = [];
            for (let row = 0; row < reshapeShape[2]; row++) {
                let rowData = [];
                for (let col = 0; col < reshapeShape[3]; col++) {
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

export default class PostProcess {
    constructor(options) {
        this.modelConfig = models[options.modelName];
        this.count = 0;
        this.lastRect = [0, 0, 0, 0]
    }
    
    run(data, img, callback, canavs) {
        let {from, to} = this.modelConfig.outputShapes;
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
            // shape: [19, 19, 25, 1],
            // reshapeShape: [19, 19, 5, 5]
            shape: from,
            reshapeShape: to
        });
        // console.log('reshape', formatData3);
        // 4.运算
        let finalData = this.handleFinal(formatData3, shape, img);
        // console.log('final', finalData);
        // 5.处理画布
        // finalData.length && handleCanvas(finalData, img);
        this.handleDiv(finalData, img, callback, canavs);
    }

    calSize(img) {
        let w1 = img.width;
        let h1 = img.height;
        let wh1 = Math.max(w1, h1);
        let factor = this.modelConfig.feedShape.fw / wh1;
        // let factor = 608.0 / wh1;
        let width = Math.round(w1 * factor);
        let height = Math.round(h1 * factor);
        return [w1, h1, width, height];
    }

    // 处理运算
    handleFinal(formatData3, shape, img) {
        let finalData = [];
        let c = shape[2];
        let [w1, h1, width, height] = this.calSize(img);
        let factorX = Math.max(width, height) / width;
        let factorY = Math.max(width, height) / height;

        let maxProb = 0.0;
        let anchors = [[1.603231, 2.094468], [6.041143, 7.080126], [2.882459, 3.518061], [4.266906, 5.178857], [9.041765, 10.66308]];

        for (let i = 0; i < shape[2]; i++) {
            for (let j = 0; j < shape[3]; j++) {
                for (let k = 0; k < anchors.length; k++) {
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
    }

    handleDiv(finalData, img, callback, canavs) {
        if (finalData.length < 1) {
            callback();
            return false;
        }
        let maxIndex = 0;
        if (finalData.length > 1) {
            for (let i = 1; i < finalData.length; i++) {
                if (finalData[i].prob > finalData[maxIndex].prob) {
                    maxIndex = i;
                }

            }
        }

        let [demoLeft, demoTop, demoWidth, demoHeight] = finalData[maxIndex];
        if (!isSimilar(this.lastRect, [demoLeft, demoTop, demoWidth, demoHeight])) {
            callback([demoWidth, demoHeight,demoLeft, demoTop], canavs);
        };
        this.lastRect = [demoLeft, demoTop, demoWidth, demoHeight];
    }

    // 处理画布
    handleCanvas(finalData, img) {
        let myCanvas = document.getElementById('myCanvas');
        let [w1, h1, width, height] = calSize(img);
        myCanvas.width = w1;
        myCanvas.height = h1;
        let ctx = myCanvas.getContext('2d');
        // ctx.drawImage(img, 0, 0, w1, h1);

        // finalData.forEach((demoArr, index) => {
        // let [demoLeft, demoTop, demoWidth, demoHeight, prob] = demoArr;
        let [demoLeft, demoTop, demoWidth, demoHeight, prob] = finalData[0];
        ctx.beginPath();
        ctx.lineWidth = 4;
        ctx.strokeStyle = 'red';
        ctx.moveTo(demoLeft, demoTop);
        ctx.lineTo(demoLeft + demoWidth, demoTop);
        ctx.lineTo(demoLeft + demoWidth, demoTop + demoHeight);
        ctx.lineTo(demoLeft, demoTop + demoHeight);
        ctx.closePath();
        ctx.stroke();
        // });
    }
}

