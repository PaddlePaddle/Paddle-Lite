/* eslint-disable */
/**
 * @file 激活函数
 * @author yangmingming
 */
// 激活函数
export default `
float prelu(float x, float p, float b) {
    float result = x;
    if (x < 0.0) {
        result = x * p;
    }
    return result;
}
`;
