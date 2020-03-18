/* eslint-disable */
/**
 * @file softmax激活函数
 * @author wangqun
 */
export default `
float softmax(float x, float p, float b) {
    float result = x;
    if (x < 0.0) {
        result = x * p;
    }
    return result;
}
`;
