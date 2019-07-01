/* eslint-disable */
/**
 * @file 公共方法, 获取[H, W]的总和
 * @author yangmingming
 */
export default `
float getRangeSumFromArrayIndex_TEXTURE_NAME(int start) {
    float result = 0.0;
    for (int i = 0; i < (width_shape_TENSOR_NAME * height_shape_TENSOR_NAME); i++) {
        vec3 pos = getTexturePosFromArrayIndex_TEXTURE_NAME(i + start);
        result += getValueFromTexturePos_TEXTURE_NAME(pos); 
    }
    return result;
}
`;
