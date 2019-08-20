/* eslint-disable */
/**
 * @file mul主函数
 */
export default `
// start函数
void main(void) {
    float res = 0.0;
    // 获取output的坐标
    ivec4 out_pos = getOutputTensorPos();
    for (int j = 0; j < width_shape_origin; j++) {
        float c = getValueFromTensorPos_counter(out_pos[0], out_pos[1], j, out_pos[3]);
        float o = getValueFromTensorPos_origin(out_pos[0], out_pos[1], out_pos[2], j);
        res += c * o;
    }
    setOutput(res);
}
`;
