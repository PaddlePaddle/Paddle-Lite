/**
 * @file 工具类
 * @author wangqun, yangmingming
 */
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

    applyFilterWinograd(data, shape) {
        const [b, c, h, w] = shape;
        let offset = 0;
        let index = 0;
        const result = new Float32Array(b * c * 16);
        // h和w是3、3
        const size2D = 9;
        for (let i = 0; i < b; i++) {
            // let index = i * c * size2D;
            for (let j = 0; j < c; j++) {
                // index += j * size2D;
                const filter = data.subarray(index, index + size2D);
                const [f11, f12, f13, f21, f22, f23, f31, f32, f33] = filter;
                const square = [
                    f11,
                    0.5 * f11 + 0.5 * f12 + 0.5 * f13,
                    0.5 * f11 - 0.5 * f12 + 0.5 * f13,
                    f13,
                    0.5 * f11 + 0.5 * f21 + 0.5 * f31,
                    0.25 * f11 + 0.25 * f12 + 0.25 * f13 + 0.25 * f21 + 0.25 * f22 + 0.25 * f23 + 0.25 * f31 + 0.25 * f32 + 0.25 * f33,
                    0.25 * f11 - 0.25 * f12 + 0.25 * f13 + 0.25 * f21 - 0.25 * f22 + 0.25 * f23 + 0.25 * f31 - 0.25 * f32 + 0.25 * f33,
                    0.5 * f13 + 0.5 * f23 + 0.5 * f33,
                    0.5 * f11 - 0.5 * f21 + 0.5 * f31,
                    0.25 * f11 + 0.25 * f12 + 0.25 * f13 - 0.25 * f21 - 0.25 * f22 - 0.25 * f23 + 0.25 * f31 + 0.25 * f32 + 0.25 * f33,
                    0.25 * f11 - 0.25 * f12 + 0.25 * f13 - 0.25 * f21 + 0.25 * f22 - 0.25 * f23 + 0.25 * f31 - 0.25 * f32 + 0.25 * f33,
                    0.5 * f13 - 0.5 * f23 + 0.5 * f33,
                    f31,
                    0.5 * f31 + 0.5 * f32 + 0.5 * f33,
                    0.5 * f31 - 0.5 * f32 + 0.5 * f33,
                    f33
                ];
                result.set(square, offset);
                offset += 16;
                index += size2D;
            }
        }
        return result;
    },

    /**
     * 获取texture形状和补0个数
     * @param shape {Array} tensor的形状
     * @return {{shape: *[], zeroNumber: number}} {Object} texture信息
     */
    getTextureInfoFromTensorShape(shape = [], isPacked = false) {
        let b = shape[0] || 1;
        let c = shape[1] || 1;
        let h = shape[2] || 1;
        let w = shape[3] || 1;
        let height = b * h;
        let width = c * w;
        let offsetX = 0;
        let offsetY = 0;
        // 安卓和ios的max texture size是4096, 改造存储空间(2bh, cw / 2)
        let exceedMax = false;
        if (height > 4096 || width > 4096) {
            height *= 2;
            width = c * (Math.ceil(w / 2));
            exceedMax = true;
        }
        if (isPacked) {
            // 紧凑布局
            height = b * c * Math.ceil(h / 2);
            width = Math.ceil(w / 2);
            offsetX = w % 2;
            offsetY = h % 2;
        }
        return {
            offsetX,
            offsetY,
            exceedMax,
            shape: [4, height, width],
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
        let data = new Float32Array(b * c * h * w * 4);
        let offset = 0;
        for (let i = 0; i < total; i++) {
            let j = (i / (c * w)) | 0;
            let k = i % (c * w);
            let b1 = j / h | 0;
            let h1 = j % h;
            let c1 = k % c;
            let w1 = k / c | 0;
            let l = b1 * (c * h * w) + c1 * (h * w) + h1 * (w) + w1;
            data[offset] = renderData.data[l];
            offset += 4;
        }
        renderData.data = data;
    }
};
