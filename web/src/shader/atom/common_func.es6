/* eslint-disable */
/**
 * @file 公共方法
 * @author yangmingming
 */
export default `
// 激活函数
float prelu(float x, float p, float b) {
    float result = x;
    if (x < 0.0) {
        result = x * p;
    }
    
    return result;
}
float relu6(float x, float threshold, float b) {
        float result = max(0.0,x);
        result = min(result,threshold);
        return result;
}
float leakyRelu(float x, float p, float b) {
    float result = max(x, x * p);
    return result;
}

float scale(float x, float p, float b) {
    float result = p * x + b;
    return result;
}

float sigmoid(float x, float y, float z) {
    float result = 1.0 / (1.0 + exp(-x));
    return result;
}

float softmax(float x, float p, float b) {
    float result = exp(x) / (10.0 * exp(x));
    return result;
}

`;

