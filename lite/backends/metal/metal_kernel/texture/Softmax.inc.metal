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

#define CONCAT2(a, b) a ## b
#define CONCAT2_(a, b) a ## _ ## b

#define FUNC(f, p) CONCAT2_(f, p)
#define VECTOR(p, n) CONCAT2(p, n)

kernel void FUNC(softmax, P)(texture2d_array<P, access::read> inTexture [[texture(0)]],
                             texture2d_array<P, access::write> outTexture [[texture(1)]],
                             constant SoftmaxParam &sp [[buffer(0)]],
                             uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) return;

    // find max value
    P max_value = inTexture.read(uint2(0, gid.y), 0)[0];
    int loop = sp.K / 4;
    int left = sp.K % 4;
    int w = inTexture.get_width();
    int h = inTexture.get_height();
    int array_size = inTexture.get_array_size();
    for (int z = 0; z < array_size; z++) {
        for (int y = 0; y < h; y++) {
            int l = 0;
            for (int x = 0; x < w; x++) {
                VECTOR(P, 4) temp_value_vector = inTexture.read(uint2(x, y), z);
                if(l< loop) {
                  for (int c = 0; c < 4; c++) {
                      max_value = max(max_value, temp_value_vector[c]);
                  }
                    l++;
                } else {
                                    for (int c = 0; c < left; c++) {
                                        max_value = max(max_value, temp_value_vector[c]);
                                    }
                }
            }
        }
    }

    // calculate sum
    VECTOR(P, 4) sum_vector = 0.0;
    for (int z = 0; z < array_size; z++) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                VECTOR(P, 4) temp_value_vector = inTexture.read(uint2(x, y), z);
                sum_vector += exp(temp_value_vector - max_value);
            }
        }
    }

    P sum_value = sum_vector[0] + sum_vector[1] + sum_vector[2] + sum_vector[3];

    // calculate output
    VECTOR(P, 4) result_vector = inTexture.read(gid.xy, gid.z);
    result_vector = exp(result_vector - max_value) / sum_value;
    outTexture.write(result_vector, gid.xy, gid.z);
}

kernel void FUNC(softmax2, P)(texture2d_array<P, access::read> inTexture [[texture(0)]],
                             texture2d_array<P, access::write> outTexture [[texture(1)]],
                             constant SoftmaxParam &sp [[buffer(0)]],
                             uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() ||
        gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size()) return;

    // find max value
    P max_value = inTexture.read(uint2(gid.x, gid.y), 0)[0];
    int w = inTexture.get_width();
    int h = inTexture.get_height();
    int array_size = inTexture.get_array_size();

    VECTOR(P, 4) max_vector = inTexture.read(uint2(gid.x, gid.y), 0);

    for(int i = 0 ; i<sp.K; i++){
      max_value = max(max_value, max_vector[i]);
    }

    // calculate sum
    VECTOR(P, 4) sum_vector = 0.0;
    VECTOR(P, 4) temp_value_vector = inTexture.read(uint2(gid.x, gid.y), gid.z);
    sum_vector += exp(temp_value_vector - max_value);

    P sum_value = 0.0;
    for(int i = 0 ; i<sp.K; i++){
       sum_value += sum_vector[i];
    }

    // calculate output
    VECTOR(P, 4) result_vector = inTexture.read(gid.xy, gid.z);
    result_vector = exp(result_vector - max_value) / sum_value;
    outTexture.write(result_vector, gid.xy, gid.z);
}

#endif
