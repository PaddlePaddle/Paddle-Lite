/* eslint-disable */
/**
 * @file 公共方法
 * @author yangmingming
 */
// TENSOR_NAME, tensor name
// 获取数组元素索引为N的元素，在tensor上的坐标ivec4(batch, channel, height, width)
export default `
iTENSOR_TYPE getTensorPosFromArrayIndex_TENSOR_NAME(int n) {
    iTENSOR_TYPE pos;
    pos[0] = n / numbers_shape_TENSOR_NAME[0];
    for (int i = 1; i < length_shape_TENSOR_NAME; i++) {
        n = int(mod(float(n), float(numbers_shape_TENSOR_NAME[i - 1])));
        pos[i] = n / numbers_shape_TENSOR_NAME[i];
    }
    return pos;
}
`;
