/* eslint-disable */
/**
 * @file 加法主函数
 * @author yangmingming
 */
export default `
// start函数
void main(void) {
    // 输出数据
    ivec4 oPos = getOutputTensorPosLIMIT_OUT();
    int index = oPos[axis];
    float o = getPixelsFromTexturePos_texture_origin(vCoord).r;
    float c = getValueFromCounter(index);
    float res = ACTIVE_FUNCTION(o + c, multi_value, bias_value);
    setOutput(res);
}
`;
