/* eslint-disable */
/**
 * @file 公共方法
 * @author yangmingming
 */
export default `
// TENSOR_TYPE, tensor坐标的类型,ivec4
// TENSOR_NAME, tensor name

// 获取tensor坐标对应数组中的索引
// uniform int numbers_shape_TENSOR_NAME[LENGTH_SHAPE_TENSOR_NAME];

int getArrayIndexFromTensorPos_TENSOR_NAME(TENSOR_TYPE tensorPos) {
    int index = 0;
    for (int i = 0; i < length_shape_TENSOR_NAME; i++) {
        index += tensorPos[i] * numbers_shape_TENSOR_NAME[i];
    }
    return index;
}
`;
