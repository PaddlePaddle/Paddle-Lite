/* eslint-disable */
/**
 * @file 公共方法
 * @author yangmingming
 * desc 根据材质坐标获取这个材质位置的值
 */
// TEXTURE_NAME, tensor name
// 获取材质中的数据
// uniform sampler2D TEXTURE_NAME;
export default `
float getValueFromTexturePos_TEXTURE_NAME(vec3 pos) {
    vec4 pixels = TEXTURE2D(TEXTURE_NAME, pos.xy);
    int d = int(pos.z);
    if (d == 0) {
        return pixels.r;
    } else if (d == 1) {
        return pixels.g;
    } else if (d == 2) {
        return pixels.b;
    }
    return pixels.a;
}
`;
