/* eslint-disable */
/**
 * @file 预设条件
 * @author yangmingming
 */
export default `
#ifdef GL_FRAGMENT_PRECISION_HIGH
    precision highp float;
    precision highp int;
#else
    precision mediump float;
    precision mediump int;
#endif
    varying vec2 vCoord;
    void setOutput(float result) {
        gl_FragColor.r = result;
    }
`;
