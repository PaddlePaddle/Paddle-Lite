/* eslint-disable */
/**
 * @file 公共方法-尾部, 方法1: 获取输出坐标
 * @author yangmingming
 */
export default `
vec2 _2d_shape_texture_out = vec2(float(width_texture_out), float(height_texture_out));
ivec4 getOutputTensorPos() {
    // 获取原始长度
    vec2 outCoord = vCoord.xy * _2d_shape_texture_out;
    int x = int(outCoord.x / float(channel_out));
    int c = int(mod(outCoord.x, float(channel_out)));
    int y = int(mod(outCoord.y, float(height_shape_out)));
    int b = int(outCoord.y / float(height_shape_out));
    return ivec4(b, c, y, x);
}
`;
