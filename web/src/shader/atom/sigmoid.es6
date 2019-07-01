/* eslint-disable */
/**
 * @file 激活函数
 * @author yangmingming
 */
// 激活函数
export default `
float sigmoid(float x, float y, float z) {
    float result = 1.0 / (1.0 + exp(-x));
    return result;
}
`;
