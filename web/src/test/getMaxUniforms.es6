/**
 * @file 获取当前环境的max uniform变量
 * @author yangmingming
 */
// uniform变量类型
const enums = {
    0x8B50: 'FLOAT_VEC2',
    0x8B51: 'FLOAT_VEC3',
    0x8B52: 'FLOAT_VEC4',
    0x8B53: 'INT_VEC2',
    0x8B54: 'INT_VEC3',
    0x8B55: 'INT_VEC4',
    0x8B56: 'BOOL',
    0x8B57: 'BOOL_VEC2',
    0x8B58: 'BOOL_VEC3',
    0x8B59: 'BOOL_VEC4',
    0x8B5A: 'FLOAT_MAT2',
    0x8B5B: 'FLOAT_MAT3',
    0x8B5C: 'FLOAT_MAT4',
    0x8B5E: 'SAMPLER_2D',
    0x8B60: 'SAMPLER_CUBE',
    0x1400: 'BYTE',
    0x1401: 'UNSIGNED_BYTE',
    0x1402: 'SHORT',
    0x1403: 'UNSIGNED_SHORT',
    0x1404: 'INT',
    0x1405: 'UNSIGNED_INT',
    0x1406: 'FLOAT'
};
export default function(gl, program) {
    // max fragment shader, 安卓是256, 桌面chrome浏览器是1024
    const result = {
        attributes: [],
        uniforms: [],
        attributeCount: 0,
        uniformCount: 0,
        maxVertexShader: gl.getParameter(gl.MAX_VERTEX_UNIFORM_VECTORS),
        maxFragmentShader: gl.getParameter(gl.MAX_FRAGMENT_UNIFORM_VECTORS)
    };
    const activeUniforms = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
    const activeAttributes = gl.getProgramParameter(program, gl.ACTIVE_ATTRIBUTES);
    // Loop through active uniforms
    for (let i = 0; i < activeUniforms; i++) {
        const uniform = gl.getActiveUniform(program, i);
        uniform.typeName = enums[uniform.type];
        result.uniforms.push(uniform);
        result.uniformCount += uniform.size;
    }

    // Loop through active attributes
    for (let i = 0; i < activeAttributes; i++) {
        const attribute = gl.getActiveAttrib(program, i);
        attribute.typeName = enums[attribute.type];
        result.attributes.push(attribute);
        result.attributeCount += attribute.size;
    }

    return result;
};
