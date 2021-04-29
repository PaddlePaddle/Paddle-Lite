/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
 http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License. */

#include <metal_stdlib>
#include "Common.metal"

using namespace metal;

kernel void elementwise_add(texture2d_array<ftype, access::read> inputX [[texture(0)]],
                            texture2d_array<ftype, access::read> inputY [[texture(1)]],
                            texture2d_array<ftype, access::write> outTexture [[texture(2)]],
                            constant ElementwiseAddParam &pm [[buffer(0)]],
                            uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) return;
		ftype4 rx, ry;
    rx = inputX.read(gid.xy, gid.z);
    if (pm.fast == 1) {
        ry = inputY.read(gid.xy, gid.z);
    } else if (pm.addByChannel == 1) {
				ry = inputY.read(uint2(gid.z, 0), 0);
    } else {
				//X数据存储坐标（GPU上的WHNC表示）
				int32_t x_xyzn[4] = {int32_t(gid.x), int32_t(gid.y), int32_t(gid.z), 0};
				//X数据存储坐标（CPU上的NHWC表示）(eg:NHWC)
				int32_t x_abcd[4];
				//X数据转换后坐标系（CPU上）(eg:NHWC->NCHW) 注意：有Y复用了这个值
				int32_t t_abcd[4];
				//Y数据存储坐标系
				int32_t y_abcd[4] = {0, 0, 0, 0};
				//Y数据存储坐标系（GPU上的WHNC表示）
				int32_t y_xyzn[4];
				//X数据转换维度
        int32_t xtrans[4] = {pm.xtrans[0], pm.xtrans[1], pm.xtrans[2], pm.xtrans[3]};
				//Y数据转换维度
        int32_t ytrans[4] = {pm.ytrans[0], pm.ytrans[1], pm.ytrans[2], pm.ytrans[3]};
				//
        int32_t yshift = 4 - pm.ylen - pm.axis;
				//利用X坐标计算Y坐标读取Y数据（Y维度适配X维度）
        for (int n = 0; n < 4; n++) {
						//ry的读取Index值
            x_xyzn[3] = n;
						//X由[WHNC-GPU] -> [NHWC-CPU]
            xyzn2abcd(pm.xdim[3], x_xyzn, x_abcd);
						//X由[NHWC-CPU] -> [NCHW-CPU]
            invtrans(xtrans, x_abcd, t_abcd);
						//Y对齐X坐标[NCHW-CPU]
            for (int k = pm.axis; k < (pm.axis + pm.ylen); k++) {
                y_abcd[yshift+k] = t_abcd[k];
            }
						//Y由[NCHW-CPU] -> [NHWC-CPU]
            trans(ytrans, y_abcd, t_abcd);
						//Y由[NHWC-CPU] -> [WHNC-GPU]
            abcd2xyzn(pm.ydim[3], t_abcd, y_xyzn);
						//读取Y
            ry[n] = inputY.read(uint2(y_xyzn[0], y_xyzn[1]), y_xyzn[2])[y_xyzn[3]];
        }
    }
		ftype4 r = rx + ry;
    outTexture.write(r, gid.xy, gid.z);
}

kernel void elementwise_sub(texture2d_array<ftype, access::read> inputX [[texture(0)]],
                            texture2d_array<ftype, access::read> inputY [[texture(1)]],
                            texture2d_array<ftype, access::write> outTexture [[texture(2)]],
                            constant ElementwiseParam &pm [[buffer(0)]],
                            uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) return;
		ftype4 rx, ry;
    rx = inputX.read(gid.xy, gid.z);
    if (pm.byChannel == 1) {
        ry = inputY.read(uint2(0, 0), gid.z);
    } else {
        ry = inputY.read(gid.xy, gid.z);
    }
		ftype4 r = rx - ry;
    outTexture.write(r, gid.xy, gid.z);
}

kernel void elementwise_mul(texture2d_array<ftype, access::read> inputX [[texture(0)]],
                            texture2d_array<ftype, access::read> inputY [[texture(1)]],
                            texture2d_array<ftype, access::write> outTexture [[texture(2)]],
                            constant ElementwiseParam &pm [[buffer(0)]],
                            uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) return;
		ftype4 rx, ry;
    rx = inputX.read(gid.xy, gid.z);
    if (pm.byChannel == 1) {
				ry = inputY.read(uint2(0, 0), gid.z);
		} else {
				ry = inputY.read(gid.xy, gid.z);
		}
		ftype4 r = rx * ry;
		outTexture.write(r, gid.xy, gid.z);
}
