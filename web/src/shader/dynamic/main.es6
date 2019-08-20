/* eslint-disable */
/**
 * @file 主函数
 * @author yangmingming
 */
export default `
// start函数
void main(void) {
    // 输出数据
    float o = getPixelsFromTexturePos_texture_origin(vCoord).r;
    float res = ACTIVE_FUNCTION(o, multi_value, bias_value);
    setOutput(res);
}
`;
