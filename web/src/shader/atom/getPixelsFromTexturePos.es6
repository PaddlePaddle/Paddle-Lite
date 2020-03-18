/* eslint-disable */
/**
 * @file 公共方法
 * @author yangmingming
 * desc 根据当前材质坐标位置获取值
 */
// 获取材质中的像素
export default `
#define getPixelsFromTexturePos_TEXTURE_NAME(pos) TEXTURE2D(TEXTURE_NAME, pos)
`;
