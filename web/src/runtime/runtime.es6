/* eslint-disable */
import Gpu from '../gpu/gpu';
import getMaxUniforms from '../test/getMaxUniforms';
/**
 * @file gpu运行时
 * @author wangqun@baidu.com, yangmingming@baidu.com
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

    getWebglVersion() {
        return this.gpu.getWebglVersion();
    },

    run(opName, opData, isRendered) {
        // console.dir(['fscode', opData.fsCode]);
        // let time = +Date.now();
        // let start = time;
        // let timeObj = {};
        if (!opData.isPass) {
            console.log('跳过当前op：' + opName);
            return this;
        }
        // 设置gpu参数
        const gpu = this.gpu;
        gpu.setOutProps(opData.tensor['out']);
        // 生成帧缓存材质
        gpu.attachFrameBuffer(opData.iLayer);
        // let end = +Date.now();
        let bufferStatus = gpu.frameBufferIsComplete();
        if (bufferStatus.isComplete) {
            // start = +Date.now();
            // timeObj['buferstatus-time'] = start - end;
            // gpu.attachShader(opData.fshader);
            gpu.setProgram(opData.program, isRendered);
            // end = +Date.now();
            // timeObj['createshader-time'] = end - start;
            // timeObj['jsTime'] = end - time;
            // statistic.push(timeObj);
            // 开始计算
            this.gpu.render(opData.renderData, opData.iLayer, isRendered);
            return this;
        } else {
            return bufferStatus.message;
        }
    },

    /**
     * 读取op计算结果, 并返回数据
     */
    read2() {
        let bufferStatus = this.gpu.frameBufferIsComplete();
        if (bufferStatus.isComplete) {

            return this.gpu.compute();
        }
        return null;
    },

    async read() {
        const pbo = this.gpu.createPBO();
        await this.gpu.createAndWaitForFence();
        // log.end('运行耗时');
        // log.start('后处理');
        // 其实这里应该有个fetch的执行调用或者fetch的输出
        // log.start('后处理-读取数据');
        // 开始读数据
        return this.gpu.downloadFoat32TensorFromBuffer(pbo);
    },

    createProgram(fsCode, outTensor) {
        const fshader = this.gpu.initShader(fsCode, 'fragment');
        const program = this.gpu.createProgram(fshader, outTensor);
        // test uniforms的个数
        // const maxUniforms = getMaxUniforms(this.gpu.gl, program);
        // alert(maxUniforms.maxFragmentShader);
        // console.table(maxUniforms.uniforms);
        return program;
    },

    // 释放资源
    dispose() {
        this.gpu.dispose();
    }
};
