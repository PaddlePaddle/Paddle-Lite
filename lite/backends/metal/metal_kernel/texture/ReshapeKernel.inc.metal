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

kernel void reshape(texture2d_array<ftype, access::read> inTexture [[texture(0)]],
										texture2d_array<ftype, access::write> outTexture [[texture(1)]],
										constant ReshapeParam &rp [[buffer(0)]],
										uint3 gid [[thread_position_in_grid]]) {
	
		if (gid.x >= outTexture.get_width() ||
				gid.y >= outTexture.get_height() ||
				gid.z >= outTexture.get_array_size()) return;
		
		//输出坐标：GPU上坐标{x,y,z,n} 此时一个位置表示4个数据 每个数据都是C通道上的数据
		int oxyzn[4] = {int(gid.x), int(gid.y), int(gid.z), 0};
		//输出位置：CPU上坐标 {z/outC, y, x, z%outC} 即为NHWC表示
		//Tensor数据转换后上传到CPU之前的数据表示的位置
		int oabcd[4];
		//输入坐标 GPU
		int ixyzn[4];
		//输入位置：CPU Tensor转换后上传到GPU前的数据表示的位置
		int iabcd[4];
		
		ftype4 r = ftype4(0.0);
		ReshapeParam lrp = rp;
		int count = lrp.odim[0] * lrp.odim[1] * lrp.odim[2] * lrp.odim[3];
		for (int n = 0; n < 4; n++) {
			oxyzn[3] = n;
			//输出C通道大小
			int oC = lrp.odim[lrp.otrans[3]];
			//GPU坐标转为NHWC坐标（即上传到GPU前的数据表示的位置）
			xyzn2abcd_4(oC, oxyzn, oabcd);
			int tabcd[4];
			//按照Tensor的转换来转换坐标 此处逻辑与Metal_image中desc_逻辑一致
			//4维有转换即Tensor的NCHW->NHWC 3维则没有
			//eg: tensor={1, 24, 208, 208} -> dim={1, 208, 208, 24}
			//eg: tensor={1, 9, 3549} -> dim={1, 1, 9, 3549}
			invtrans(lrp.otrans, oabcd, tabcd);
			//CPU上的NHWC位置转为CPU上Tensor的NHWC位置
			int index = abcd2index(lrp.odim, tabcd);
			if (index < count) {
				//下面逻辑与上述逻辑一致 相反的过程而已
				index2abcd(lrp.idim, index, tabcd);
				trans(lrp.itrans, tabcd, iabcd);
				int iC = lrp.idim[lrp.itrans[3]];
				abcd2xyzn_4(iC, iabcd, ixyzn);
				r[n] = inTexture.read(uint2(ixyzn[0], ixyzn[1]), ixyzn[2])[ixyzn[3]];
			} else {
				r[n] = 0;
			}
		}
		outTexture.write(r, gid.xy, gid.z);
}
