/* eslint-disable */
/**
 * @file 公共方法
 * @author yangmingming
 */
// TEXTURE_NAME, 材质名称
// WIDTH_TEXTURE_NAME_VALUE, 材质宽度
// HEIGHT_TEXTURE_NAME_VALUE, 材质高度

// 获取数组元素索引为N的元素，在texture上的坐标
// const int width_TEXTURE_NAME = WIDTH_TEXTURE_NAME_VALUE;
// const int height_TEXTURE_NAME = HEIGHT_TEXTURE_NAME_VALUE;
export default `
vec3 getTexturePosFromArrayIndex_TEXTURE_NAME(int n) {
    vec3 pos;
    pos.z = mod(float(n), 4.0);
    n /= 4;
    int y = n / width_TEXTURE_NAME;
    float width = float(width_TEXTURE_NAME);
    float x = mod(float(n), width);
    pos.x = x / width;
    pos.y = float(y) / float(height_TEXTURE_NAME);
    return pos;
}
`;
