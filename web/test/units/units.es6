import Utils from '../../src/utils/utils';
import Gpu from '../../src/gpu/gpu';
import Matrix from '../../src/utils/dims';
import axios from 'axios';
let qs = require('qs');

/**
 * @file gpu运行时
 * @author wangqun
 *
 */
// v_shader.c表示计算容器
const VSHADER = require('../../src/shader/v_shader.c');

export default {

    /**
     * 初始化op
     * @param {Object} opts 运行时参数，包含el：canvas，dim: 256
     * @return {Object} this 实例对象
     */

    async init(opts = {}, opShader) {
        const gpu = this.gpu = new Gpu(opts);
        if (gpu.isFloatingTexture()) {
            let texture = gpu.makeTexure(WebGLRenderingContext.FLOAT, null);
            let framebuffer  = gpu.attachFrameBuffer(texture);
            let bufferStatus = gpu.frameBufferIsComplete();
            if (bufferStatus.isComplete) {
                console.log(bufferStatus.isComplete);
                // 获取shader
                const vshaderCode = await Utils.loadShader(VSHADER);
                let fshaderCode = await Utils.loadShader(opShader);
                fshaderCode = Utils.populateData('conv2d', fshaderCode, opts);
                gpu.create(vshaderCode, fshaderCode);
                return this;
            } else {
                return bufferStatus.message;
            }

        } else {
            return null;
        }

    },

    /**
     * 计算op
     * @param bufferA
     * @param bufferB
     */
    compute(bufferA, bufferB, type) {
        this.gpu.render(bufferA, bufferB, type);
    },

    /**
     * 读取op计算结果, 并返回数据
     */
    read() {
        return this.gpu.compute();
    },

    // 生成feed数据
    feed(pixelData, size) {
        return Utils.shapeData(pixelData, size);
    },

    // mock生成shapeB的数据
    mockShapeB(shapeA, shapeB) {
        return Utils.mock(shapeA, shapeB);
    },

    // mock origin 1 * 5 * 5
    mockOrigin() {
        return new Matrix({
            sx: 5,
            sy: 5,
            depth: 4
        });
    },

    // mock filter 1 * 3 * 3
    mockFilter() {
        return new Float32Array([1.0, 1.0, 0.0, 0.0, -2.0, 0.0, 1.0, -3.0, 1.0]);
    },

    // 更新op
    updateOp(name) {
        // this.gpu.updateShader();
    },

    // get paddle mobile result
    getResult(name, input, output) {



        if (name) {
            let that = this;
            axios.defaults.withCredentials = false;
            axios.defaults.headers = {
                'Content-type': 'application/x-www-form-urlencoded'
            }
            axios.post('http://yq01-paddle-mobile.epc.baidu.com:8088/uniTest', qs.stringify({
                name: name,
                input: JSON.stringify(input, function (key, value) {
                    if (value.constructor === Float32Array) {
                        return that.formatData(value);
                    }else {
                        return that.formatData(value);
                    }
                }),
                output: JSON.stringify(output, function (key, value) {
                    return that.formatData(value);
                })
            },{ indices: false }))
                .then(function (response) {
                    if (response.status === 200) {
                        that.displayResult(response.data);
                    }
                    console.log(response);
                })
                .catch(function (error) {
                    console.log(error);
                });
        }

    },

    displayResult(res) {
        if (res.name) {
            let assert = (res.correct == 1? 'Pass' : 'Not pass');
            let passCls = (res.correct == 1? 'pass' : 'no-pass');
            if (res.correct === 1) {

                let unitHtml = '<li class="unit-li"><div class="unit-li-name">' + res.name + '</div>' +
                    '<div class="unit-li-assert">' + assert + '</div>'
                '</li>';
                let oli = document.createElement('li');
                oli.innerHTML = unitHtml;
                document.getElementById('paddle-web-unit-list').appendChild(oli);
            }
            else if (res.correct === 0) {
                let serverData = res.server_data;
                let unitHtml = '<li class="unit-li"><div class="unit-li-name">' + res.name + '</div>' +
                    '<div class="unit-li-assert ' + passCls + '">' + assert + '</div>' +
                    '<div class="unit-li-diff"><p>' + serverData + '</p></div>'
                    '</li>';
                let oli = document.createElement('li');
                oli.innerHTML = unitHtml;
                document.getElementById('paddle-web-unit-list').appendChild(oli);
            }
        }
    },

    formatData(list) {
        if (list.constructor === Float32Array) {
            return '[' + list.toString() + ']';
        }
        else {
            return list;
        }
    },

// 释放资源
    dispose() {
        this.gpu.dispose();
    }
};
