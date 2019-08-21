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

ivec4 getOutputTensorPosLimit() {
    // 获取原始长度
    vec2 outCoord = vCoord.xy * _2d_shape_texture_out;
    float offsetY = floor(outCoord.y / float(height_shape_out));
    int x = int(outCoord.x / float(channel_out));
    if (mod(offsetY, 2.0) > 0.0) {
        x += int(ceil(float(width_shape_out) / 2.0));
    }
    int y = int(mod(outCoord.y, float(height_shape_out)));
    int c = int(mod(outCoord.x, float(channel_out)));
    int b = int(outCoord.y / float(2 * height_shape_out));
    return ivec4(b, c, y, x);
}

ivec4 getOutputPackedTensorPos() {
    // 获取原始长度
    vec2 outCoord = vCoord.xy * _2d_shape_texture_out;
    int height = height_shape_out + offset_y_out;
    int x = int(outCoord.x);
    int c = int(outCoord.y / float(height / 2));
    int y = int(mod(outCoord.y, float(height / 2)));
    int b = 0;
    return ivec4(b, c, y, x);
}
`;
