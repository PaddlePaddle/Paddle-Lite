/* eslint-disable */
/**
 * @file conv2d的配置文件
 * @author yangmingming
 */
export default {
    dep: [
        {
            func: 'getValueFromTensorPos',
            conf: {
                TENSOR_NAME: 'origin'
            }
        },
        {
            func: 'getValueFromTensorPos',
            conf: {
                TENSOR_NAME: 'filter'
            }
        }
    ],
    conf: [
        'LENGTH_SHAPE_FILTER',
        'WIDTH_SHAPE_FILTER',
        'HEIGHT_SHAPE_FILTER',
        'WIDTH_TEXTURE_FILTER',
        'HEIGHT_TEXTURE_FILTER',
        'CHANNEL_FILTER',

        'WIDTH_SHAPE_ORIGIN',
        'HEIGHT_SHAPE_ORIGIN',
        'LENGTH_SHAPE_ORIGIN',
        'WIDTH_TEXTURE_ORIGIN',
        'HEIGHT_TEXTURE_ORIGIN',
        'CHANNEL_ORIGIN',

        'WIDTH_SHAPE_OUT',
        'HEIGHT_SHAPE_OUT',
        'WIDTH_TEXTURE_OUT',
        'HEIGHT_TEXTURE_OUT',
        'CHANNEL_OUT',
        'OFFSET_Y_OUT',

        'STRIDE_HORIZONTAL',
        'STRIDE_VERTICAL',
        'PAD_LEFT',
        'PAD_TOP',
        'DILATION_HORIZONTAL',
        'DILATION_VERTICAL',
        'GROUPS',
        'MULTI_VALUE',
        'BIAS_VALUE',
        'ACTIVE_FUNCTION'
    ],
    input: [
        // {
        //     tensor: 'filter',
        //     variable: 'numbers_shape',
        //     setter: 'uniform1iv',
        //     type: 'uniform'
        // },
        {
            tensor: 'filter',
            variable: 'texture',
            setter: 'initTexture',
            type: 'texture'
        },
        {
            tensor: 'origin',
            variable: 'texture',
            setter: 'initTexture',
            type: 'texture'
        }
        // {
        //     tensor: 'origin',
        //     variable: 'numbers_shape',
        //     setter: 'uniform1iv',
        //     type: 'uniform'
        // },
        // {
        //     tensor: 'out',
        //     variable: 'numbers_shape',
        //     setter: 'uniform1iv',
        //     type: 'uniform'
        // }
    ]
};
