/* eslint-disable */
/**
 * @file 预设条件, webgl 2.0版本
 * @author yangmingming
 */
export default `#version 300 es

#ifdef GL_FRAGMENT_PRECISION_HIGH
    precision highp float;
    precision highp int;
#else
    precision mediump float;
    precision mediump int;
#endif

// 顶点shader透传的材质坐标
    in vec2 vCoord;
    out vec4 outColor;
    void setOutput(float result) {
        outColor.r = result;
    }
`;
