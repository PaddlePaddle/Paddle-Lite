/* eslint-disable */
/**
 * @file softmax主函数
 * @author yangmingming
 */
export default `
// start函数
void main(void) {
    // 输出数据
    ivec4 oPos = getOutputTensorPos();
    float o = getValueFromTensorPos_origin(oPos);
    // 归一化数据
    vec4 scale = getPixelsFromTexturePos_texture_scale(vec2((float(int(oPos.g)) + 0.5) / float(width_texture_scale), 0.0));
    float x = (o - scale[3]) / sqrt(scale[2] + epsilon);
    float res = scale[0] * x + scale[1];
    gl_FragColor.r = res;
}
`;
