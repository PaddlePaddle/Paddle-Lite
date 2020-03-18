import ops from './ops';
/**
 * @file 工厂类，生成fragment shader
 * @author wangqun
 */
export default class Factory {
    constructor(opts) {
        this.defaultOpts = Object.assign({}, opts);
        this.webglVersion = 2;
        this.texture2d = 'texture';
    }

    setWebglVersion(vs = 0) {
        this.webglVersion = vs;
        if (vs === 1) {
            this.texture2d = 'texture2D';
        }
    }

    buildShader(opName, data) {
        let result = '';
        result = this.buildPrefix(opName);
        result += this.buildCommon(opName);
        result += this.buildOp(opName);
        data.texture2d = this.texture2d;
        result = this.populateData(result, data);
        return result;
    }

    buildPrefix(opName) {
        if (this.webglVersion === 1) {
            return ops.common.prefix;
        }
        return ops.common.prefix2;
    }

    buildCommon(opName) {
        return ops.common.params + ops.common.func;
    }

    buildOp(opName) {
        let code = ops.ops[opName].params;
        // 依赖的方法
        let atoms = ops.atoms;
        let confs = ops.ops[opName].confs;
        let dep = confs.dep || [];
        dep.map(item => {
            let func = item.func;
            let data = item.conf;
            let snippet = atoms[func];
            code += this.populateData(snippet, data);
        });
        // suffix
        code += this.buildSuffix(opName);
        // main方法
        code += ops.ops[opName].func;
        return code;
    }

    buildSuffix(opName) {
        return ops.common.suffix;
    }

    populateData(result, data) {
        let code = result;
        for (let key in data) {
            code = code.replace(new RegExp(key.toUpperCase(), 'g'),
                ((typeof data[key]) === 'undefined') ? 1 : data[key]);
        }
        return code;
    }

    getOpConfs() {
        const opsConfs = {};
        for (let key in ops.ops) {
            if (ops.ops.hasOwnProperty(key)) {
                opsConfs[key] = ops.ops[key].confs.input;
            }
        }
        return opsConfs;
    }
}

