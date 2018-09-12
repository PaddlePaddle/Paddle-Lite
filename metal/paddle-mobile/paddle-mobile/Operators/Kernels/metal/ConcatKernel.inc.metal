#ifdef P

#define CONCAT2(a, b) a ## b
#define CONCAT2_(a, b) a ## _ ## b
#define CONCAT3_(a, b, c) a ## _ ## b ## _ ## c

#define FUNC(f, r, p) CONCAT3_(f, r, p)
#define VECTOR(p, n) CONCAT2(p, n)
#define FUNC_R(f, r) CONCAT2_(f, r)

kernel void FUNC(concat, R, P)(texture2d_array<P, access::read> in0 [[texture(0)]],
                   texture2d_array<P, access::read> in1 [[texture(1)]],
                   texture2d_array<P, access::read> in2 [[texture(2)]],
                   texture2d_array<P, access::read> in3 [[texture(3)]],
                   texture2d_array<P, access::read> in4 [[texture(4)]],
                   texture2d_array<P, access::read> in5 [[texture(5)]],
                   texture2d_array<P, access::read> inx [[texture(6)]],
                   texture2d_array<P, access::write> out [[texture(7)]],
                   constant ConcatParam & pm [[buffer(0)]],
                   uint3 gid [[thread_position_in_grid]]) {
   ConcatParam cp = pm;
   int xyzn[4] = {int(gid.x), int(gid.y), int(gid.z), 0}, abcd[4], oxyzn[4];
   VECTOR(P, 4) r;
   for (int i = 0; i < 4; i++) {
     xyzn[3] = i;
#if R == 4
     xyzn2abcd_4(cp.odim[3], xyzn, abcd);
#else
     FUNC_R(xyzn2abcd, R)(xyzn, abcd);
#endif
     int k = abcd[cp.axis] - cp.offset;
     int j = 0;
     if (k < 0) {
       r[i] = inx.read(gid.xy, gid.z)[i];
     } else {
       for (; j < 6; j++) {
         if (k < cp.vdim[j]) {
           break;
         }
         k -= cp.vdim[j];
       }
       int ta = cp.odim[cp.axis];
       abcd[cp.axis] = k;
       cp.odim[cp.axis] = cp.vdim[j];
#if R == 4
       abcd2xyzn_4(cp.odim[3], abcd, oxyzn);
#else
       FUNC_R(abcd2xyzn, R)(abcd, oxyzn);
#endif
       cp.odim[cp.axis] = ta;
       switch (j) {
         case 0: r[i] = in0.read(uint2(oxyzn[0], oxyzn[1]), oxyzn[2])[oxyzn[3]]; break;
         case 1: r[i] = in1.read(uint2(oxyzn[0], oxyzn[1]), oxyzn[2])[oxyzn[3]]; break;
         case 2: r[i] = in2.read(uint2(oxyzn[0], oxyzn[1]), oxyzn[2])[oxyzn[3]]; break;
         case 3: r[i] = in3.read(uint2(oxyzn[0], oxyzn[1]), oxyzn[2])[oxyzn[3]]; break;
         case 4: r[i] = in4.read(uint2(oxyzn[0], oxyzn[1]), oxyzn[2])[oxyzn[3]]; break;
         case 5: r[i] = in5.read(uint2(oxyzn[0], oxyzn[1]), oxyzn[2])[oxyzn[3]]; break;
       }
     }
   }
   out.write(r, gid.xy, gid.z);
}
#endif
