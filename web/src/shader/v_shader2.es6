/* eslint-disable */
/**
 * @file 顶点文件,webgl 2.0
 * @author wangqun
 * @desc  顶点坐标系转换，适配webgl2
 */
export default `#version 300 es
in vec4 position;
out vec2 vCoord;
void main() {
    vCoord.x = (position.x + 1.0) / 2.0;
    vCoord.y = (position.y + 1.0) / 2.0;
    gl_Position = position;
}
`;
