/* eslint-disable */
/**
 * @file image，feed 获取图像相关输入
 * @author wangqun@baidu.com
 */
export default class imageFeed {
    constructor() {
        this.fromPixels2DContext = document.createElement('canvas').getContext('2d');
        this.fromPixels2DContext2 = document.createElement('canvas').getContext('2d');
        this.defaultWidth = 224;
        this.defaultHeight = 224;
        this.minPixels = 225;
        this.pixels = '';
        this.defaultParams = {
            gapFillWith: '#000',
            std: [1, 1, 1]
        };
    };

    /**
     * 处理图像方法
     * @param inputs
     */
    process(inputs) {
        const input = inputs.input;
        const mode = inputs.mode;
        const channel = inputs.channel;
        const rotate = inputs.rotate;
        const params = {
            ...this.defaultParams,
            ...inputs.params
        };
        let output = [];
        if (!this.result) {
            const [b, c, h, w] = params.targetShape;
            // 计算确定targetShape所需Float32Array占用空间
            this.result = new Float32Array(h * w * c);
        }
        output = this.fromPixels(input, params);
        return output;
    };

    /**
     * crop图像&重新设定图片tensor形状
     * @param shape
     */
    reshape(imageData, opt, scaleSize) {
        const {sw, sh} = scaleSize;
        const {width, height} = opt;
        const hPadding = Math.ceil((sw - width) / 2);
        const vPadding = Math.ceil((sh - height) / 2);

        let data = imageData.data;
        // channel RGB
        let red = [];
        let green = [];
        let blue = [];
        // 平均数
        let mean = opt.mean;
        // 标准差
        let std = opt.std;
        // 考虑channel因素获取数据
        for (let i = 0; i < data.length; i += 4) {

            let index = i / 4;
            let vIndex = Math.floor(index / sw);
            let hIndex = index - (vIndex * sw) - 1;
            if (hIndex >= hPadding && hIndex < (hPadding + width) &&
                vIndex >= vPadding && vIndex < (vPadding + height)) {
                red.push(((data[i] / 255) - mean[0]) / std[0]); // red
                green.push(((data[i + 1] / 255) - mean[1]) / std[1]); // green
                blue.push(((data[i + 2] / 255) - mean[2]) / std[2]); // blue
            }
        }
        // 转成 GPU 加速 NCHW 格式
        let tmp = green.concat(blue);
        return red.concat(tmp);
    };

    /**
     * 全部转rgb * H * W
     * @param shape
     */
    allReshapeToRGB(imageData, opt, scaleSize) {
        const {sw, sh} = scaleSize;
        const [b, c, h, w] = opt.targetShape;
        let data = imageData.data || imageData;
        let mean = opt.mean;
        let dataLength = data.length;
        // let result = new Float32Array(dataLength * 3);
        let result = this.result;
        // let offsetR = 0;
        // let offsetG = dataLength / 4;
        // let offsetB = dataLength / 2;
        let offset = 0;
        let size = h * w;
        // h w c
        for (let i = 0; i < h; ++i) {
            let iw = i * w;
            for (let j = 0; j < w; ++j) {
                let iwj = iw + j;
                for (let k = 0; k < c; ++k) {
                    let a = iwj * 4 + k;
                    result[offset++] = (data[a] - mean[k]) / 256;
                }
            }
        }
        return result;
    };

    /**
     * 根据scale缩放图像
     * @param image
     * @param params
     * @return {Object} 缩放后的尺寸
     */
    reSize(image, params) {
        // 原始图片宽高
        const width = this.pixelWidth;
        const height = this.pixelHeight;
        // 缩放后的宽高
        let sw = width;
        let sh = height;
        // 最小边缩放到scale
        if (width < height) {
            sw = params.scale;
            sh = Math.round(sw * height / width);
        } else {
            sh = params.scale;
            sw = Math.round(sh * width / height);
        }
        this.fromPixels2DContext.canvas.width = sw;
        this.fromPixels2DContext.canvas.height = sh;
        this.fromPixels2DContext.drawImage(
            image, 0, 0, sw, sh);
        this.setInputCanvas(image);
        return {sw, sh};
    };


    /**
     * 缩放成目标尺寸并居中
     */
    fitToTargetSize(image, params, center) {
        // 目标尺寸
        const targetWidth = params.targetSize.width;
        const targetHeight = params.targetSize.height;
        this.fromPixels2DContext.canvas.width = targetWidth;
        this.fromPixels2DContext.canvas.height = targetHeight;
        this.fromPixels2DContext.fillStyle = params.gapFillWith;
        this.fromPixels2DContext.fillRect(0, 0, targetHeight, targetWidth);
        // 缩放后的宽高
        let sw = targetWidth;
        let sh = targetHeight;
        let x = 0;
        let y = 0;
        // target的长宽比大些 就把原图的高变成target那么高
        if (targetWidth / targetHeight * this.pixelHeight / this.pixelWidth >= 1) {
            sw = Math.round(sh * this.pixelWidth / this.pixelHeight);
            x = Math.floor((targetWidth - sw) / 2);
        }
        // target的长宽比小些 就把原图的宽变成target那么宽
        else {
            sh = Math.round(sw * this.pixelHeight / this.pixelWidth);
            y = Math.floor((targetHeight - sh) / 2);
        }
        // console.log(x, y, sw, sh);
        if (center) {
            this.fromPixels2DContext.drawImage(
                image, x, y, sw, sh);
        }
        else {
            this.fromPixels2DContext.drawImage(
                image, 0, 0, sw, sh);
            // currentPic = this.fromPixels2DContext.canvas.toDataURL();
        }
        this.setInputCanvas(image);
        // window.currentPic = this.fromPixels2DContext.canvas;// test only, demele me
        // document.getElementById('p-c').appendChild(this.fromPixels2DContext.canvas);// test only, demele me
        return {sw: targetWidth, sh: targetHeight};
    }

    /**
     * 设置原始video画布
     * @param image 原始video
     */
    setInputCanvas(image) {
        // 原始图片宽高
        const width = this.pixelWidth;
        const height = this.pixelHeight;
        // 画布设置
        this.fromPixels2DContext2.canvas.width = width;
        this.fromPixels2DContext2.canvas.height = height;
        this.fromPixels2DContext2.drawImage(image, 0, 0, width, height);
    }

    /**
     * 获取图像内容
     * @param pixels
     * @returns {Uint8ClampedArray}
     */
    getImageData(pixels, scaleSize) {
        const {sw, sh} = scaleSize;
        // 复制画布上指定矩形的像素数据
        let vals = this.fromPixels2DContext
            .getImageData(0, 0, sw, sh);
        // crop图像
        // const width = pixels.width;
        // const height = pixels.height;
        return vals;
    };

    /**
     * 计算灰度图
     * @param imageData
     * @returns {*}
     */
    grayscale (imageData) {
        let data = imageData.data;

        for (let i = 0; i < data.length; i += 4) {
            // 3 channel 灰度处理无空间压缩
            let avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
            data[i] = avg; // red
            data[i + 1] = avg; // green
            data[i + 2] = avg; // blue
        }
        return data;
    };

    fromPixels(pixels, opt) {
        let data;
        // 原始video画布数据
        let data2;
        let scaleSize;
        if (pixels instanceof HTMLImageElement || pixels instanceof HTMLVideoElement) {
            this.pixelWidth = pixels.naturalWidth || pixels.width;
            this.pixelHeight = pixels.naturalHeight || pixels.height;
            if (opt.scale) { // 兼容以前的，如果有scale就是短边缩放到scale模式
                scaleSize = this.reSize(pixels, opt);
                data = this.getImageData(opt, scaleSize);
                data2 = this.fromPixels2DContext2.getImageData(0, 0, this.pixelWidth, this.pixelHeight);
            }
            else if (opt.targetSize) { // 如果有targetSize，就是装在目标宽高里的模式
                scaleSize = this.fitToTargetSize(pixels, opt);
                data = this.getImageData(opt, scaleSize);
                data2 = this.fromPixels2DContext2.getImageData(0, 0, this.pixelWidth, this.pixelHeight);
            }
        }

        if (opt.gray) {
            data = grayscale(data);
        }

        if (opt.reShape) {
            data = this.reshape(data, opt, scaleSize);
        }

        if (opt.targetShape) {
            data = this.allReshapeToRGB(data, opt, scaleSize);
        }
        
        return [{data: data, shape: opt.shape || opt.targetShape, name: 'image', canvas: data2}];
    }
}
/* eslint-enable */
