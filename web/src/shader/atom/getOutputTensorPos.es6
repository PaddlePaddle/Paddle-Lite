/* eslint-disable */
/**
 * @file 公共方法
 * @author yangmingming
 * @desc 获取输出tensor的坐标
 */
export default `
ivec4 getOutputTensorPos() {
    // 获取原始长度
    vec2 outCoord = moveTexture2PosToReal_texture_out(vCoord.xy);
    // 材质体系转tensor体系坐标位置
    int x = int(outCoord.x / float(channel_out));
    int c = int(mod(outCoord.x, float(channel_out)));
    int y = int(mod(outCoord.y, float(height_shape_out)));
    int b = int(outCoord.y / float(height_shape_out));
    return ivec4(b, c, y, x);
}
`;
