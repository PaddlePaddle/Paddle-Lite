/* eslint-disable */
import VSHADER from '../shader/v_shader';
import VSHADER2 from '../shader/v_shader2';
/**
 * @file gpu运算
 * @author wangqun@baidu.com, yangmingming@baidu.com
 */
const CONF = {
    alpha: false,
    antialias: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: false,
    depth: false,
    stencil: false,
    failIfMajorPerformanceCaveat: true
};
const MAX_WAIT = 100;
export default class gpu {
    constructor(opts = {}) {
        // 版本, 默认webgl version 2.0
        this.version = 2;
        this.opts = opts;
        opts.width_raw_canvas = Number(opts.width_raw_canvas) || 512;
        opts.height_raw_canvas = Number(opts.height_raw_canvas) || 512;
        const canvas = opts.el ? opts.el : document.createElement('canvas');
        canvas.addEventListener('webglcontextlost', evt => {
            evt.preventDefault();
            console.log('webgl context is lost~');
        }, false);
        let gl = canvas.getContext('webgl2', CONF);
        if (!!gl) {
            // 开启float32
            this.version = 2;
            this.textureFloat = gl.getExtension('EXT_color_buffer_float');
            this.internalFormat = gl.R32F;
            this.textureFormat = gl.RED;
            this.downloadInternalFormat = gl.RGBA32F;
        } else {
            gl = canvas.getContext('webgl', CONF) || canvas.getContext('experimental-webgl', CONF);
            this.version = 1;
            this.internalFormat = gl.RGBA;
            this.textureFormat = gl.RGBA;
            this.downloadInternalFormat = gl.RGBA;
            if (!gl) {
                this.version = 0;
                alert('当前环境创建webgl context失败');
            } else {
                // 开启扩展
                this.textureFloat  = gl.getExtension('OES_texture_float');
                console.log('float extension is started or not? ' + !!this.textureFloat);
            }
        }
        // 关闭相关功能
        gl.disable(gl.DEPTH_TEST);
        gl.disable(gl.STENCIL_TEST);
        gl.disable(gl.BLEND);
        gl.disable(gl.DITHER);
        gl.disable(gl.POLYGON_OFFSET_FILL);
        gl.disable(gl.SAMPLE_COVERAGE);
        gl.enable(gl.SCISSOR_TEST);
        gl.enable(gl.CULL_FACE);
        gl.cullFace(gl.BACK);
        this.gl = gl;
        this.initCache();
        // 同步查看次数
        this.waits = 0;

        console.log('WebGl版本是 ' + this.version);
        console.log('MAX_TEXTURE_SIZE is ' + gl.getParameter(gl.MAX_TEXTURE_SIZE));
        console.log('MAX_TEXTURE_IMAGE_UNITS is ' + gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS));
    }

    getWebglVersion() {
        return this.version;
    }

    initCache() {
        // 运行次数
        this.times = 0;
        const gl = this.gl;
        // 顶点数据
        let vertices = new Float32Array([
            -1.0,  1.0, 0.0, 1.0,
            -1.0, -1.0, 0.0, 0.0,
            1.0,  1.0, 1.0, 1.0,
            1.0, -1.0, 1.0, 0.0]);
        this.vertexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
        // shader
        this.vertexShader = null;
        // 生成vertextShader
        this.initShader(this.version === 2 ? VSHADER2 : VSHADER);
        this.fragmentShader = null;
        // 上一个texture
        this.prevTexture = null;
        // 当前op输出texture
        this.currentTexture = null;
        // 帧缓存
        this.frameBuffer = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.frameBuffer);
        // 计算texture cache
        this.cacheTextures = {};
        this.uniformLocations = {};
        // texture buffer
        this.outTextures = [];
        // pbo
        this.pbo = gl.createBuffer();
    }

    runVertexShader(program) {
        const gl = this.gl;
        let aPosition = gl.getAttribLocation(program, 'position');
        // Turn on the position attribute
        gl.enableVertexAttribArray(aPosition);
        // Bind the position buffer.
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
        gl.vertexAttribPointer(aPosition, 2, gl.FLOAT, false, 16, 0);
    }

    setOutProps(opts) {
        this.width_shape_out = opts.width_shape || 1;
        this.height_shape_out = opts.height_shape || 1;
        this.width_texture_out = opts.width_texture || 1;
        this.height_texture_out = opts.height_texture || 1;
        this.channel = opts.channel || 0;
        this.total_shape = opts.total_shape || 0;
    }

    isFloatingTexture() {
        return (this.textureFloat !== null);
    }

    createProgram(fshader, out) {
        const gl = this.gl;
        const program = gl.createProgram();
        gl.attachShader(program, this.vertexShader);
        gl.attachShader(program, fshader);
        gl.linkProgram(program);
        // 生成output的texture缓存
        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        gl.texImage2D(gl.TEXTURE_2D, // Target, matches bind above.
            0,             // Level of detail.
            this.downloadInternalFormat,       // Internal format.
            out.width_texture,
            out.height_texture,
            0,             // Always 0 in OpenGL ES.
            gl.RGBA,       // Format for each pixel.
            gl.FLOAT,          // Data type for each chanel.
            null);
        gl.bindTexture(gl.TEXTURE_2D, null);
        this.outTextures.push(texture);
        return program;
    }

    setProgram(program, isRendered) {
        const gl = this.gl;
        gl.useProgram(program);
        this.program = program;
        if (!isRendered) {
            this.runVertexShader(program);
        }
    }

    attachShader(fshader) {
        const gl = this.gl;
        // let index = this.textureBufferIndex % 2;
        // const program = this.programs[index];
        // this.program = program;
        const program = this.program;
        // if (this.times < 2) {
        //     gl.attachShader(program, this.vertexShader);
        // }
        this.textureBufferIndex = (this.textureBufferIndex + 1) >= 2 ? 0 : 1;
        if (!!this.fragmentShader) {
            gl.detachShader(program, this.fragmentShader);
        }
        this.gl.attachShader(program, fshader);
        this.fragmentShader = fshader;
        gl.linkProgram(program);
        if (this.times++ === 0) {
            gl.useProgram(program);
            this.runVertexShader();
        }
    }

    create(vshaderCode, fshaderCode) {
        let gl = this.gl;
        if (this.program) {
            this.dispose();
        }
        // 创建 & 绑定程序对象
        let program = this.program = gl.createProgram();
        // 创建&绑定vertex&frament shader
        this.initShader(vshaderCode);
        this.fragmentShader = this.initShader(fshaderCode, 'fragment');
        this.gl.attachShader(program, this.vertexShader);
        this.gl.attachShader(program, this.fragmentShader);
        gl.linkProgram(program);
        gl.useProgram(program);

        let aPosition = gl.getAttribLocation(program, 'position');
        // Turn on the position attribute
        gl.enableVertexAttribArray(aPosition);
        // Bind the position buffer.
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
        gl.vertexAttribPointer(aPosition, 2, gl.FLOAT, false, 16, 0);
    }

    /**
     * 初始化shader
     * @param code shader代码
     * @param type shader类型
     * @return {object} 初始化成功返回shader
     */
    initShader(code, type = 'vertex') {
        const shaderType = type === 'vertex' ? this.gl.VERTEX_SHADER : this.gl.FRAGMENT_SHADER;
        let shader;
        if (type === 'vertex' && this.vertexShader) {
            shader = this.vertexShader;
        } else {
            shader = this.gl.createShader(shaderType);
            if (type === 'vertex') {
                this.vertexShader = shader;
            }
            this.gl.shaderSource(shader, code);
            this.gl.compileShader(shader);
            if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
                throw new Error("compile: " + this.gl.getShaderInfoLog(shader));
            }
        }

        return shader;
    }

    /**
     * 更新fragment shader
     * @param code shader代码
     * @return {boolean} 更新成功过返回true
     */
    updateShader(code) {
        this.gl.useProgram(this.program);
        // 删除fragment shader
        if (this.fragmentShader) {
            this.gl.detachShader(this.program, this.fragmentShader);
            this.gl.deleteShader(this.fragmentShader);
            // 删除texture
            this.gl.deleteTexture(this.texture);
        }
        // 更新
        this.fragmentShader = this.initShader(code, 'fragment');
        return true;
    }

    /**
     * 创建并绑定framebuffer, 之后attach a texture
     * @param {WebGLTexture} texture 材质
     * @returns {WebGLFramebuffer} The framebuffer
     */
    attachFrameBuffer(iLayer) {
        this.prevTexture = this.currentTexture;
        // this.currentTexture = this.textureBuffer[this.textureBufferIndex % 2];
        // this.textureBufferIndex = (this.textureBufferIndex + 1) >= 2 ? 0 : 1;
        this.currentTexture = this.outTextures[iLayer];
        console.log('this.currentTexture', this.currentTexture);
        const gl = this.gl;
        gl.framebufferTexture2D(gl.FRAMEBUFFER, // The target is always a FRAMEBUFFER.
            gl.COLOR_ATTACHMENT0, // We are providing the color buffer.
            gl.TEXTURE_2D, // This is a 2D image texture.
            this.currentTexture, // The texture.
            0 // 0, we aren't using MIPMAPs
        );
        gl.viewport(
            0,
            0,
            this.width_texture_out,
            this.height_texture_out
        );
        gl.scissor(
            0,
            0,
            this.width_texture_out,
            this.height_texture_out
        );
        return this.frameBuffer;
    }

    // 帧缓存检测
    frameBufferIsComplete() {
        let gl = this.gl;
        let message;
        let status;
        let value;

        status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);

        switch (status)
        {
            case gl.FRAMEBUFFER_COMPLETE:
                message = "Framebuffer is complete.";
                value = true;
                break;
            case gl.FRAMEBUFFER_UNSUPPORTED:
                message = "Framebuffer is unsupported";
                value = false;
                break;
            case gl.FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
                message = "Framebuffer incomplete attachment";
                value = false;
                break;
            case gl.FRAMEBUFFER_INCOMPLETE_DIMENSIONS:
                message = "Framebuffer incomplete (missmatched) dimensions";
                value = false;
                break;
            case gl.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
                message = "Framebuffer incomplete missing attachment";
                value = false;
                break;
            default:
                message = "Unexpected framebuffer status: " + status;
                value = false;
        }
        return {isComplete: value, message: message};
    }

    /**
     * 初始化材质
     * @param {int} index 材质索引
     * @param {string} tSampler 材质名称
     * @param {Object} bufferData 数据
     * @param {boolean} isRendered 是否已运行过
     */
    initTexture(index, item, iLayer, isRendered) {
        const gl = this.gl;
        let texture;
        if (!item.data) {
            texture = this.prevTexture;
        } else {
            // texture = gl.createTexture();
            if (isRendered && (iLayer > 0 || (iLayer === 0 && item.tensor !== 'origin'))) {
                const tData = this.cacheTextures['' + iLayer];
                texture = tData[item.variable + '_' + item.tensor];
            } else {
                texture = gl.createTexture();
                if (index === 0) {
                    this.cacheTextures['' + iLayer] = this.cacheTextures['' + iLayer] || {};
                }
                this.cacheTextures['' + iLayer][item.variable + '_' + item.tensor] = texture;
            }
        }
        gl.activeTexture(gl[`TEXTURE${index}`]);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        if (item.data && (!isRendered || (isRendered && iLayer === 0 && item.tensor === 'origin'))) {
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texImage2D(gl.TEXTURE_2D,
                0,
                this.internalFormat,
                item.width_texture,
                item.height_texture,
                0,
                this.textureFormat,
                gl.FLOAT,
                item.data,
                0);
        }
    }

    getUniformLoc(name, ilayer, isRendered) {
        if (isRendered) {
            return this.uniformLocations['' + ilayer][name];
        }
        let loc = this.gl.getUniformLocation(this.program, name);
        if (loc === null) throw `getUniformLoc ${name} err`;
        // 缓存
        this.uniformLocations['' + ilayer] = this.uniformLocations['' + ilayer] || {};
        this.uniformLocations['' + ilayer][name] = loc;
        return loc;
    }

    // 生成帧缓存的texture
    makeTexure(type, data, opts = {}) {
        const gl = this.gl;
        let index = this.textureBufferIndex % 2;
        let texture = this.textureBuffer[index];
        gl.bindTexture(gl.TEXTURE_2D, texture);

        // Pixel format and data for the texture
        gl.texImage2D(gl.TEXTURE_2D, // Target, matches bind above.
            0,             // Level of detail.
            gl.RGBA,       // Internal format.
            opts.width_texture_out || this.width_texture_out,
            opts.height_texture_out || this.height_texture_out,
            0,             // Always 0 in OpenGL ES.
            gl.RGBA,       // Format for each pixel.
            type,          // Data type for each chanel.
            data);         // Image data in the described format, or null.
        // Unbind the texture.
        // gl.bindTexture(gl.TEXTURE_2D, null);
        this.attachFrameBuffer();

        return texture;
    }

    render(data = [], iLayer = 0, isRendered = false) {
        const gl = this.gl;
        let that = this;
        let textureIndex = 0;
        data.forEach(item => {
            if (item.type === 'texture') {
                that.initTexture(textureIndex, item, iLayer, isRendered);
                gl.uniform1i(that.getUniformLoc(item.variable + '_' + item.tensor, iLayer, isRendered), textureIndex++);
            }
            else if (item.type === 'uniform') {
                gl[item.setter](that.getUniformLoc(item.variable + '_' + item.tensor, iLayer, isRendered), item.data);
            }
        });
        // gl.clearColor(.0, .0, .0, 1);
        // gl.clear(gl.COLOR_BUFFER_BIT);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    createPBO() {
        const gl2 = this.gl;
        const buffer = this.pbo;
        gl2.bindBuffer(gl2.PIXEL_PACK_BUFFER, buffer);
        const sizeBytes = 4 * 4 * this.width_texture_out * this.height_texture_out;
        gl2.bufferData(gl2.PIXEL_PACK_BUFFER, sizeBytes, gl2.STREAM_READ);
        gl2.readPixels(0, 0, this.width_texture_out, this.height_texture_out, gl2.RGBA, gl2.FLOAT, 0);
        gl2.bindBuffer(gl2.PIXEL_PACK_BUFFER, null);
        return buffer;
    }

    downloadFoat32TensorFromBuffer(buffer) {
        const gl2 = this.gl;
        const size = 4 * this.width_texture_out * this.height_texture_out;
        const pixels = new Float32Array(size);
        gl2.bindBuffer(gl2.PIXEL_PACK_BUFFER, buffer);
        gl2.getBufferSubData(gl2.PIXEL_PACK_BUFFER, 0, pixels);
        gl2.bindBuffer(gl2.PIXEL_PACK_BUFFER, null);
        // log.start('后处理-readloop');
        // let result = [];
        // let offset = 0;
        // for (let h = 0; h < this.height_texture_out; h++) {
        //     // 纪录第1和2行数据
        //     let temp1 = [];
        //     let temp2 = [];
        //     for (let w = 0; w < this.width_texture_out; w++) {
        //         temp1.push(pixels[offset]);
        //         temp1.push(pixels[offset + 1]);
        //         temp2.push(pixels[offset + 2]);
        //         temp2.push(pixels[offset + 3]);
        //         offset += 4;
        //     }
        //     result = result.concat(temp1);
        //     result = result.concat(temp2);
        // }
        let result = [];
        for (let i = 0; i < this.width_texture_out * this.height_texture_out; i++) {
            result.push(pixels[4 * i]);
        }
        // const result = Array.prototype.slice.call(pixels);
        // console.dir(['result', result]);
        // log.end('后处理-readloop');
        return result;
    }

    getWebglError(status) {
        const gl2 = this.gl;
        switch (status) {
            case gl2.NO_ERROR:
                return 'NO_ERROR';
            case gl2.INVALID_ENUM:
                return 'INVALID_ENUM';
            case gl2.INVALID_VALUE:
                return 'INVALID_VALUE';
            case gl2.INVALID_OPERATION:
                return 'INVALID_OPERATION';
            case gl2.INVALID_FRAMEBUFFER_OPERATION:
                return 'INVALID_FRAMEBUFFER_OPERATION';
            case gl2.OUT_OF_MEMORY:
                return 'OUT_OF_MEMORY';
            case gl2.CONTEXT_LOST_WEBGL:
                return 'CONTEXT_LOST_WEBGL';
            default:
                return `Unknown error code ${status}`;
        }
    }

    createAndWaitForFence() {
        const gl2 = this.gl;
        const isFenceEnabled = (gl2.fenceSync !== null);
        let isFencePassed = () => true;
        if (isFenceEnabled) {
            const sync = gl2.fenceSync(gl2.SYNC_GPU_COMMANDS_COMPLETE, 0);
            gl2.flush();
            isFencePassed = () => {
                const status = gl2.clientWaitSync(sync, 0, 0);
                return status === gl2.ALREADY_SIGNALED ||
                    status === gl2.CONDITION_SATISFIED;
            };
        }
        return new Promise(resolve => {
            this.pollItem(isFencePassed, resolve);
        });
    }

    pollItem(isDone, resolveFn) {
        const fn = () => {
            if (isDone()) {
                resolveFn();
                return;
            }
            setTimeout(fn, 1);
        };
        fn();
    }

    compute() {
        let gl = this.gl;
        // log.start('后处理-readinside');
        const tt = +Date.now();
        let pixels = new Float32Array(this.width_texture_out * this.height_texture_out * 4);
        // gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
        const tt2 = +Date.now();
        gl.readPixels(0, 0, this.width_texture_out, this.height_texture_out, gl.RGBA, gl.FLOAT, pixels, 0);
        // console.log('本次读取数据时间是' + (+Date.now() - tt2)+ ',' + (tt2 - tt));
        // log.end('后处理-readinside');
        // log.start('后处理-readloop');
        let result = [];
        for (let i = 0; i < this.width_texture_out * this.height_texture_out; i++) {
            result.push(pixels[4 * i]);
        }
        // log.end('后处理-readloop');
        return result;
    }

    dispose() {
        const gl = this.gl;
        // this.cacheTextures.forEach(texture => {
        //     gl.deleteTexture(texture);
        // });
        this.cacheTextures = {};
        this.programs.forEach(program => {
            gl.detachShader(program, this.vertexShader);
            gl.deleteShader(this.vertexShader);
            gl.deleteProgram(program);
        });
        this.programs = [];
    }
}
