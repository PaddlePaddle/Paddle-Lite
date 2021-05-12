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

#ifdef P

kernel void FUNC2_(pool, P)(texture2d_array<P, access::read> inTexture [[texture(0)]],
                            texture2d_array<P, access::write> outTexture [[texture(1)]],
                            constant PoolParam &pm [[buffer(0)]],
                            uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) return;
    int xmin = gid.x * pm.strideX - pm.paddingX;
    int xmax = min(xmin + pm.ksizeX, int(inTexture.get_width()));
    xmin = max(xmin, 0);
    int ymin = gid.y * pm.strideY - pm.paddingY;
    int ymax = min(ymin + pm.ksizeY, int(inTexture.get_height()));
    ymin = max(ymin, 0);
    
    VECTOR(P, 4) r = 0;
    if (pm.poolType == 0) {
        r = inTexture.read(uint2(xmin, ymin), gid.z);
        for (int x = xmin; x < xmax; x++) {
            for (int y = ymin; y < ymax; y++) {
                r = fmax(r, inTexture.read(uint2(x, y), gid.z));
            }
        }
    } else if (pm.poolType == 1) {
        for (int x = xmin; x < xmax; x++) {
            for (int y = ymin; y < ymax; y++) {
                r += inTexture.read(uint2(x, y), gid.z);
            }
        }
        int count = pm.exclusive? (xmax - xmin) * (ymax - ymin) : (pm.ksizeY * pm.ksizeX);
            VECTOR(P, 4) div = count > 0 ? 1.f / count : 0.0;
                r *= div;
    }
    outTexture.write(r, gid.xy, gid.z);
}

#endif
