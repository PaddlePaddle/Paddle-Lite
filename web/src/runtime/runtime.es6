/* eslint-disable */
import Gpu from '../gpu/gpu';
/**
 * @file gpu运行时
 * @author yangmingming
 *
 */
export default {
    /**
     * 初始化, 生成gpu实例
     * @param {Object} opts 运行时参数，包含el：canvas，dim: 256
     * @return {Object} this 实例对象
     */
    init(opts = {}) {
        const gpu = this.gpu = new Gpu(opts);
        if (gpu.isFloatingTexture()) {
            return this;
        } else {
            return null;
        }
    },

    run(opName, opData) {
        let time = +Date.now();
        let start = time;
        let timeObj = {};
        if (!opData.isPass) {
            console.log('跳过当前op：' + opName);
            return this;
        }
        // 设置gpu参数
        const gpu = this.gpu;
        gpu.setOutProps(opData.tensor['out']);
        // 生成帧缓存材质
        gpu.makeTexure(WebGLRenderingContext.FLOAT, null);
        let end = +Date.now();
        let bufferStatus = gpu.frameBufferIsComplete();
        if (bufferStatus.isComplete) {
            start = +Date.now();
            timeObj['buferstatus-time'] = start - end;
            gpu.attachShader(opData.fshader);
            end = +Date.now();
            timeObj['createshader-time'] = end - start;
            timeObj['jsTime'] = end - time;
            statistic.push(timeObj);
            // 开始计算
            this.gpu.render(opData.renderData);
            return this;
        } else {
            return bufferStatus.message;
        }
    },

    /**
     * 读取op计算结果, 并返回数据
     */
    read() {
        return this.gpu.compute();
    },

    createFragmentShader(fsCode) {
        return this.gpu.initShader(fsCode, 'fragment');
    },

    // 释放资源
    dispose() {
        this.gpu.dispose();
    }
};
