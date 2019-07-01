/* eslint-disable */
/**
 * @file 公共方法
 * @author yangmingming
 */
// TEXTURE_NAME, tensor name
// 获取材质中的像素
// uniform sampler2D TEXTURE_NAME;
export default `
#define getPixelsFromTexturePos_TEXTURE_NAME(pos) texture2D(TEXTURE_NAME, pos)
`;
