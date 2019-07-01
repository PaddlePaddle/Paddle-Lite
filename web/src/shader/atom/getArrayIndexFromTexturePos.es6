/* eslint-disable */
/**
 * @file 公共方法
 * @author yangmingming
 */
// TEXTURE_NAME, texture name
// WIDTH_TEXTURE_NAME_VALUE, texture的宽度

// 获取材质元素在数组中的索引
// const int width_TEXTURE_NAME = WIDTH_TEXTURE_NAME_VALUE;
export default `
int getArrayIndexFromTexturePos_TEXTURE_NAME(vec3 pos) {
    int x = int(floor(pos.x));
    int y = int(floor(pos.y));
    int d = int(floor(pos.z));
    return (width_TEXTURE_NAME * y + x) * 4 + d;
}
`;
