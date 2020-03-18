/* eslint-disable */
/**
 * @file 公共方法
 * @author yangmingming
 * desc 坐标转化
 */
// TEXTURE_NAME, 材质name
// 材质坐标转化成真实尺寸坐标
export default `

// vec2 moveTexture2PosToReal_TEXTURE_NAME(vec2 v) {
//     return v * _2d_shape_TEXTURE_NAME;
//     // vec2 v2;
//     // v2.x = v.x * float(width_TEXTURE_NAME);
//     // v2.y = v.y * float(height_TEXTURE_NAME);
//     // return v2;
// }

vec2 moveTexture2PosToReal_TEXTURE_NAME(vec2 v) {
    vec2 v2;
    v2.x = v.x * float(width_TEXTURE_NAME);
    v2.y = v.y * float(height_TEXTURE_NAME);
    return v2;
}
`;
