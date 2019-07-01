/* eslint-disable */
/**
 * @file 加法参数
 * @author yangmingming
 */
export default `
    // 输入数据
    const int axis = AXIS;
    // const int total_shape_counter = TOTAL_SHAPE_COUNTER;
    uniform float data_counter[TOTAL_SHAPE_COUNTER];
    uniform sampler2D texture_origin;
    float getValueFromCounter(int index) {
        for (int i = 0; i < TOTAL_SHAPE_COUNTER; i++) {
            if (i == index) {
                return data_counter[i];
            }
        }
        return 0.0;
    }
`;
