/* eslint-disable */
/**
 * @file 公共方法
 * @author yangmingming
 *
 */
export default `

int getArrayIndexFromTensorPos_TENSOR_NAME(TENSOR_TYPE tensorPos) {
    int index = 0;
    for (int i = 0; i < length_shape_TENSOR_NAME; i++) {
        index += tensorPos[i] * numbers_shape_TENSOR_NAME[i];
    }
    return index;
}
`;
