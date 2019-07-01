/**
 * @file 工具类
 * @author yangmingming
 */
/* eslint-disable */
export default {
    // todo: 适用2维矩阵乘法，以后实现通用版本
    getReshapeInPaddle(inputShape = [], counterShape = [], outShape = []) {
        let total = inputShape.reduce((all, num) => all * num);
        if (outShape.length === 1) {
            return [1, total];
        } else {
            return [outShape[0], total / outShape[0]];
        }
    },

    getBroadcastShapeInPaddle(shapeA= [], shapeB = [], axis = 1) {
        // todo: 简易版本，以后需要实现一个通用版本
        let bigger = shapeA;
        let result = shapeB;
        if (shapeA.length - shapeB.length < 0) {
            bigger = shapeB;
            result = shapeA;
        }
        return result.concat(bigger.slice(axis));
    },

    getBroadcastDims(inShape = [], outShape = []) {
        const inRank = inShape.length;
        const dims = [];
        for (let i = 0; i < inRank; i++) {
            const dim = inRank - 1 - i;
            const a = inShape[dim] || 1;
            const b = outShape[outShape.length - 1 - i] || 1;
            if (b > 1 && a === 1) {
                dims.unshift(dim);
            }
        }
        return dims;
    },

    getBroadcastShape(shapeA = [], shapeB = []) {
        const result = [];
        const max = Math.max(shapeA.length, shapeB.length);
        for (let i = 0; i < max; i++) {
            let a = shapeA[shapeA.length - i - 1];
            if (a === null) {
                a = 1;
            }
            let b = shapeB[shapeB.length - i - 1];
            if (b === null) {
                b = 1;
            }
            if (a === 1) {
                result.unshift(b);
            } else if (b === 1) {
                result.unshift(a);
            } else if (a !== b) {
                return null;
            } else {
                result.unshift(a);
            }
        }
        return result;
    },

    /**
     * 获取texture形状和补0个数
     * @param shape {Array} tensor的形状
     * @return {{shape: *[], zeroNumber: number}} {Object} texture信息
     */
    getTextureInfoFromTensorShape(shape = []) {
        let b = shape[0];
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];
        return {
            shape: [4, b * h, c * w],
            zeroNumber: 0
        };
    },

    // 获取数组中的最大值和索引
    getMaxItem(datas = []) {
        let max = Math.max.apply(null, datas);
        let index = datas.indexOf(max);
        return {value: max, index};
    },

    // 压缩
    async loadShader(name) {
        let shader = await fetch(this.getShaderFile(name));
        return shader.text();
    },

    getShaderFile(url) {
        // todo: 根据脚手架获取shader文件
        const aa = url.split('/');
        let length = aa.length;
        return '/' + aa[length - 1];
    },

    img2texture(renderData = {}) {
        const {height_texture, width_texture, shape} = renderData;
        const total = height_texture * width_texture * 4;
        const b = shape[0];
        const c = shape[1];
        const h = shape[2];
        const w = shape[3];
        const data = [];
        for (let i = 0; i < total; i++) {
            let j = Math.floor(i / (c * w));
            let k = Math.floor(i % (c * w));
            let b1 = Math.floor(j / h);
            let h1 = Math.floor(j % h);
            let c1 = Math.floor(k % c);
            let w1 = Math.floor(k / c);
            let l = b1 * (c * h * w) + c1 * (h * w) + h1 * (w) + w1;
            data.push(renderData.data[l]);
            data.push(0);
            data.push(0);
            data.push(0);
        }
        renderData.data = new Float32Array(data);
    }
};
/* eslint-enable */
