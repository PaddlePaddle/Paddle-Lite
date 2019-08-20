/* eslint-disable */
/**
 * @file 公共方法
 * @author yangmingming
 */

export default `
int getArrayIndexFromTexturePos_TEXTURE_NAME(vec3 pos) {
    int x = int(floor(pos.x));
    int y = int(floor(pos.y));
    int d = int(floor(pos.z));
    return (width_TEXTURE_NAME * y + x) * 4 + d;
}
`;
