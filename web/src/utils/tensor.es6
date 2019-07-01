/* eslint-disable */
import Utils from './utils';
/**
 * @file Tensor类
 * @author yangmingming
 */
export default class Tensor {
    constructor(opts = {}) {
        this.opts = opts;
        // 设置tensor名字
        this.name = opts.name;
        // tensor的形状
        let shape = this.shape = opts.shape;
        // 原始数据个数
        this.total = shape.reduce((all, num) => all * num);
        // 图像tensor是否带有batch
        if (opts.needBatch && shape.length < 4) {
            let batch = [];
            for (let i = 0; i < (4 - shape.length); i++) {
                batch.push(1);
            }
            shape = batch.concat(shape);
            this.shape = shape;
        }
        // 获取转换到texture后的信息
        let {zeroNumber, shape: shape_texture} = Utils.getTextureInfoFromTensorShape(shape);
        this.shape_texture = shape_texture;
        // tensor数据
        let data = [];
        if (opts.data && opts.data.length) {
            if (!opts.notCompressed) {
                let b = shape[0];
                let c = shape[1];
                let h = shape[2];
                let w = shape[3];
                for (let i = 0; i < opts.data.length; i++) {
                    let j = Math.floor(i / (c * w));
                    let k = Math.floor(i % (c * w));
                    let b1 = Math.floor(j / h);
                    let h1 = Math.floor(j % h);
                    let c1 = Math.floor(k % c);
                    let w1 = Math.floor(k / c);
                    let l = b1 * (c * h * w) + c1 * (h * w) + h1 * (w) + w1;
                    data.push(opts.data[l]);
                    data.push(0);
                    data.push(0);
                    data.push(0);
                }
            } else {
                // batchnorm的scale
                this.shape_texture = [4, 1, this.total / 4];
                data = [].concat(opts.data);
            }

            this.data = new Float32Array(data);
            // 清理缓存
            opts.data = null;
        }
    }

    /**
     * 获取数组下标, shape例子[M, W, H, D]
     * @param pos {Array} tensor坐标索引
     * @return {Number} tensor数据
     */
    getValue(pos = []) {
        let p = [].concat(pos);
        let len = p.length;
        let sLen = this.shape.length;
        // 补齐
        for (let i = 0; i < (sLen - len); i++) {
            p.unshift(0);
        }
        let index = 0;
        for (let i = 0; i < sLen; i++) {
            index += p[i] * this.shapeNumbers[i];
        }
        return this.data[index];
    }

    get width_texture() {
        let length = this.shape_texture.length;
        return this.shape_texture[length - 1];
    }

    get height_texture() {
        let length = this.shape_texture.length;
        return this.shape_texture[length - 2];
    }

    get width_shape() {
        let length = this.shape.length;
        return this.shape[length - 1];
    }

    get height_shape() {
        let length = this.shape.length;
        return this.shape[length - 2];
    }

    get channel() {
        let length = this.shape.length;
        if (length >= 3) {
            return this.shape[length - 3];
        }
        return 0;
    }

    get length_shape() {
        return this.shape.length || 0;
    }

    /**
     * 获取shape对应的个数
     * @return {Array} 和shape长度相等的对应个数
     */
    get numbers_shape() {
        let numbers = [];
        let sLen = this.shape.length;
        for (let i = 0; i < (sLen - 1); i++) {
            let number = this.shape.slice(i + 1).reduce((total, num) => total * num);
            numbers.push(number);
        }
        // 和shape长度保持一致
        numbers.push(1);
        return numbers;
    }

    get total_shape() {
        return this.total;
    }

    dispose() {
        if (this.data) {
            this.data = null;
        }
    }
}
/* eslint-enable */
