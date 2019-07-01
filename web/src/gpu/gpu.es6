/* eslint-disable */
import VSHADER from '../shader/v_shader';
/**
 * @file gpu运算
 * @author yangmingming
 */
export default class gpu {
    constructor(opts = {}) {
        this.opts = opts;
        opts.width_raw_canvas = Number(opts.width_raw_canvas) || 512;
        opts.height_raw_canvas = Number(opts.height_raw_canvas) || 512;
        let canvas = opts.el ? opts.el : document.createElement('canvas');
        canvas.width = opts.width_raw_canvas;
        canvas.height = opts.height_raw_canvas;
        this.gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        this.gl.viewport(0, 0, canvas.width, canvas.height);
        // Attempt to activate the extension, returns null if unavailable
        this.textureFloat  = this.gl.getExtension('OES_texture_float');
        // this.setOutProps();
        this.initCache();
        console.log('float extension is started or not? ' + !!this.textureFloat);
        console.log('WebGl版本是 ' + this.gl.getParameter(this.gl.SHADING_LANGUAGE_VERSION));
    }

    initCache() {
        // 运行次数
        this.times = 0;
        const gl = this.gl;
        // 缓存每个op的texture
        // this.textures = [];
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
        this.initShader(VSHADER);
        this.fragmentShader = null;
        // 上一个texture
        this.prevTexture = null;
        // 当前op输出texture
        this.currentTexture = null;
        // 帧缓存
        this.frameBuffer = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.frameBuffer);
        // 计算texture cache, 最多3个
        this.cacheTextures = [gl.createTexture(), gl.createTexture(), gl.createTexture()];
        // texture buffer
        this.textureBuffer = [gl.createTexture(), gl.createTexture()];
        // program
        this.programs = [gl.createProgram(), gl.createProgram()];
        this.program = this.programs[0];
        this.textureBufferIndex = 0;
        for (let i = 0; i < 2; i++) {
            gl.bindTexture(gl.TEXTURE_2D, this.textureBuffer[i]);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.bindTexture(gl.TEXTURE_2D, null);
        }
    }

    runVertexShader() {
        const gl = this.gl;
        let aPosition = gl.getAttribLocation(this.program, 'position');
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
        this.total_shape = opts.total_shape || 0;
    }

    isFloatingTexture() {
        return (this.textureFloat !== null);
    }

    attachShader(fshader) {
        const gl = this.gl;
        let index = this.textureBufferIndex % 2;
        const program = this.programs[index];
        this.program = program;
        if (this.times < 2) {
            gl.attachShader(program, this.vertexShader);
        }
        this.textureBufferIndex = (this.textureBufferIndex + 1) >= 2 ? 0 : 1;
        this.gl.attachShader(program, fshader);
        gl.linkProgram(program);
        gl.useProgram(program);
        if (this.times++ < 2) {
            this.runVertexShader();
        }
        if (!!this.fragmentShader) {
            const cache = this.programs[(index + 1) % 2];
            gl.detachShader(cache, this.fragmentShader);
            // gl.deleteShader(this.fragmentShader);
        }
        this.fragmentShader = fshader;
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
    attachFrameBuffer(opts = {}) {
        this.prevTexture = this.currentTexture;
        this.currentTexture = this.textureBuffer[this.textureBufferIndex % 2];
        // this.textureBufferIndex = (this.textureBufferIndex + 1) >= 2 ? 0 : 1;
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
            opts.width_texture_out || this.width_texture_out,
            opts.height_texture_out || this.height_texture_out
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
     * 更新材质
     * @param {WebGLTexture}    texture 材质对象
     * @param {number}          type    材质类型. FLOAT, UNSIGNED_BYTE, etc.
     * @param {Float32Array[]}        data    材质数据
     */
    refreshTexture(texture, type, data) {
        const gl = this.gl;
        // Bind the texture so the following methods effect it.
        gl.bindTexture(gl.TEXTURE_2D, texture);

        // Replace the texture data
        gl.texSubImage2D(gl.TEXTURE_2D, // Target, matches bind above.
            0,             // Level of detail.
            0,             // xOffset
            0,             // yOffset
            this.opts.width_raw_canvas,  // Width - normalized to s.
            this.opts.height_raw_canvas, // Height - normalized to t.
            gl.RGBA,       // Format for each pixel.
            type,          // Data type for each chanel.
            data);         // Image data in the described format.

        // Unbind the texture.
        gl.bindTexture(gl.TEXTURE_2D, null);

        return texture;
    }

    /**
     * 初始化材质
     * @param {int} index 材质索引
     * @param {string} tSampler 材质名称
     * @param {Object} bufferData 数据
     */
    initTexture(index, item) {
        const gl = this.gl;
        let texture;
        if (!item.data) {
            texture = this.prevTexture;
        } else {
            // texture = gl.createTexture();
            texture = this.cacheTextures[index];
            // this.textures.push(texture);
        }
        gl.activeTexture(gl[`TEXTURE${index}`]);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        if (item.data) {
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, item.width_texture || this.opts.width_raw_canvas,
                item.height_texture || this.opts.height_raw_canvas, 0,
                gl.RGBA, gl.FLOAT, item.data, 0);
        }
    }

    getUniformLoc(name) {
        let loc = this.gl.getUniformLocation(this.program, name);
        if (loc === null) throw `getUniformLoc ${name} err`;
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

    render(data = []) {
        const gl = this.gl;
        let textureIndex = 0;
        // 输入数据
        data.forEach(item => {
            if (item.type === 'texture') {
                this.initTexture(textureIndex, item);
                gl.uniform1i(this.getUniformLoc(item.variable + '_' + item.tensor), textureIndex++);
            } else if (item.type === 'uniform') {
                gl[item.setter](this.getUniformLoc(item.variable + '_' + item.tensor), item.data);
            }
        });
        // gl.clearColor(.0, .0, .0, 1);
        // gl.clear(gl.COLOR_BUFFER_BIT);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    compute() {
        let gl = this.gl;
        let pixels = new Float32Array(this.width_texture_out * this.height_texture_out * 4);
        // gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
        gl.readPixels(0, 0, this.width_texture_out, this.height_texture_out, gl.RGBA, gl.FLOAT, pixels, 0);

        let result = [];
        for (let i = 0; i < this.width_texture_out * this.height_texture_out; i++) {
            result.push(pixels[4 * i]);
        }
        return result;
    }

    dispose() {
        const gl = this.gl;
        this.cacheTextures.forEach(texture => {
            gl.deleteTexture(texture);
        });
        this.cacheTextures = [];
        this.programs.forEach(program => {
            gl.detachShader(program, this.vertexShader);
            gl.deleteShader(this.vertexShader);
            gl.deleteProgram(program);
        });
        this.programs = [];
    }
}
