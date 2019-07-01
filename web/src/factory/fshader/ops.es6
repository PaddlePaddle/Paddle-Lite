/* eslint-disable */
import common_params from '../../shader/atom/common_params';
import common_func from '../../shader/atom/common_func';
import prefix from '../../shader/atom/prefix';
import suffix from '../../shader/atom/suffix';
import ivec56 from '../../shader/atom/type_ivec56';

import conv2d_params from '../../shader/conv2d/params';
import conv2d_func from '../../shader/conv2d/main';
import conv2d_conf from '../../shader/conv2d/conf';
import dynamic_params from '../../shader/dynamic/params';
import dynamic_func from '../../shader/dynamic/main';
import dynamic_conf from '../../shader/dynamic/conf';
import pool2d_params from '../../shader/pool2d/params';
import pool2d_func from '../../shader/pool2d/main';
import pool2d_conf from '../../shader/pool2d/conf';
import elementwise_add_params from '../../shader/elementwise_add/params';
import elementwise_add_func from '../../shader/elementwise_add/main';
import elementwise_add_conf from '../../shader/elementwise_add/conf';
import mul_params from '../../shader/mul/params';
import mul_func from '../../shader/mul/main';
import mul_conf from '../../shader/mul/conf';
import softmax_params from '../../shader/softmax/params';
import softmax_func from '../../shader/softmax/main';
import softmax_conf from '../../shader/softmax/conf';
import batchnorm_params from '../../shader/batchnorm/params';
import batchnorm_func from '../../shader/batchnorm/main';
import batchnorm_conf from '../../shader/batchnorm/conf';

import getArrayIndexFromTensorPos from '../../shader/atom/getArrayIndexFromTensorPos';
import getArrayIndexFromTexturePos from '../../shader/atom/getArrayIndexFromTexturePos';
import getTensorPosFromArrayIndex from '../../shader/atom/getTensorPosFromArrayIndex';
import getTexturePosFromArrayIndex from '../../shader/atom/getTexturePosFromArrayIndex';
import getValueFromTexturePos from '../../shader/atom/getValueFromTexturePos';
import getValueFromTensorPos from '../../shader/atom/getValueFromTensorPos';
import moveTexture2PosToReal from '../../shader/atom/moveTexture2PosToReal';
import getPixelsFromTexturePos from '../../shader/atom/getPixelsFromTexturePos';
import getRangePowSumFromArrayIndex from '../../shader/atom/getRangePowSumFromArrayIndex';
import getRangeSumFromArrayIndex from '../../shader/atom/getRangeSumFromArrayIndex';
import sigmoid from '../../shader/atom/sigmoid';
import prelu from '../../shader/atom/prelu';
import scale from '../../shader/atom/scale';
import softmax from '../../shader/atom/softmax';
/**
 * @file op文件
 * @author yangmingming
 */

export default {
    common: {
        params: common_params,
        func: common_func,
        prefix,
        suffix,
        ivec56
    },
    ops: {
        conv2d: {
            params: conv2d_params,
            func: conv2d_func,
            confs: conv2d_conf
        },
        dynamic: {
            params: dynamic_params,
            func: dynamic_func,
            confs: dynamic_conf
        },
        pool2d: {
            params: pool2d_params,
            func: pool2d_func,
            confs: pool2d_conf
        },
        elementwise_add: {
            params: elementwise_add_params,
            func: elementwise_add_func,
            confs: elementwise_add_conf
        },
        mul: {
            params: mul_params,
            func: mul_func,
            confs: mul_conf
        },
        relu: {
            params: dynamic_params,
            func: dynamic_func,
            confs: dynamic_conf
        },
        scale: {
            params: dynamic_params,
            func: dynamic_func,
            confs: dynamic_conf
        },
        softmax: {
            params: softmax_params,
            func: softmax_func,
            confs: softmax_conf
        },
        batchnorm: {
            params: batchnorm_params,
            func: batchnorm_func,
            confs: batchnorm_conf
        }
    },
    atoms: {
        getArrayIndexFromTensorPos,
        getArrayIndexFromTexturePos,
        getTensorPosFromArrayIndex,
        getTexturePosFromArrayIndex,
        getValueFromTexturePos,
        getValueFromTensorPos,
        moveTexture2PosToReal,
        getPixelsFromTexturePos,
        getRangeSumFromArrayIndex,
        getRangePowSumFromArrayIndex,
        sigmoid,
        prelu,
        scale,
        softmax
    }
};
