/* eslint-disable */
/**
 * @file 顶点文件
 * @author wangqun
 * @desc  顶点坐标系转换，适配webgl1
 */
export default `
attribute vec4 position;
varying vec2 vCoord;
void main() {
    vCoord.x = (position.x + 1.0) / 2.0;
    vCoord.y = (position.y + 1.0) / 2.0;
    gl_Position = position;
}
`;
