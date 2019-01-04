//
//  PoolKernel.inc.metal
//  paddle-mobile
//
//  Created by liuRuiLong on 2018/12/29.
//  Copyright © 2018 orange. All rights reserved.
//

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
  int ymin = gid.y * pm.strideX - pm.paddingX;
  int ymax = min(ymin + pm.ksizeX, int(inTexture.get_height()));
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
    r /= (xmax - xmin) * (ymax - ymin);
  }
  outTexture.write(r, gid.xy, gid.z);
}

#endif
