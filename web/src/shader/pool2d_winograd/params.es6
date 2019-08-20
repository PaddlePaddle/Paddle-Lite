/* eslint-disable */
/**
 * @file pool2d参数文件
 */
export default `
// 常量
// 池化大小
const int width_shape_pool = KSIZE_X;
const int height_shape_pool = KSIZE_Y;
const int type_pool = TYPE_POOL;
// 输入数据
const int width_shape_origin = WIDTH_SHAPE_ORIGIN;
const int height_shape_origin = HEIGHT_SHAPE_ORIGIN;
const int length_shape_origin = LENGTH_SHAPE_ORIGIN;
const int width_texture_origin = WIDTH_TEXTURE_ORIGIN;
const int height_texture_origin = HEIGHT_TEXTURE_ORIGIN;
const int channel_origin = CHANNEL_ORIGIN;
const int offset_x_origin = OFFSET_X_ORIGIN;
const int offset_y_origin = OFFSET_Y_ORIGIN;


// 计算相关
// 拆分步长
const int stride_h = STRIDES_X;
const int stride_v = STRIDES_Y;
// padding的数目
const int padLeft = PADDINGS_X;
const int padTop = PADDINGS_Y;


// uniform变量
uniform sampler2D texture_origin;
`;
