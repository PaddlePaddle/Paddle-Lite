#ifdef P

#define CONCAT2(a, b) a ## b
#define CONCAT2_(a, b) a ## _ ## b
#define CONCAT3_(a, b, c) a ## _ ## b ## _ ## c
#define CONCAT4_(a, b, c, d) a ## _ ## b ## _ ## c ## _ ## d
#define CONCAT5_(a, b, c, d, e) a ## _ ## b ## _ ## c ## _ ## d ## _ ## e

#define FUNC(f, r, n, v, p) CONCAT5_(f, r, n, v, p)
#define VECTOR(p, n) CONCAT2(p, n)
#define FUNC_R(f, r) CONCAT2_(f, r)

#if V == VX
#define VV x
#elif V == VY
#define VV y
#elif V == VZ
#define VV z
#else
#define VV normal
#endif

kernel void FUNC(split, R, N, VV, P)(texture2d_array<P, access::read> input [[texture(0)]],
                                 texture2d_array<P, access::write> out1 [[texture(1)]],
                                 texture2d_array<P, access::write> out2 [[texture(2)]],
#if N >= 3
                                 texture2d_array<P, access::write> out3 [[texture(3)]],
#endif
#if N >= 4
                                 texture2d_array<P, access::write> out4 [[texture(4)]],
#endif
                                 constant SplitParam &sp [[buffer(0)]],
                                 uint3 gid [[thread_position_in_grid]]) {

  VECTOR(P, 4) r = input.read(gid.xy, gid.z);
#if V == VY
  int y = gid.y - sp.offset;
  if (y < sp.vdim[0]) {
    out1.write(r, gid.xy, gid.z);
  } else {
    y -= sp.vdim[0];
    if (y < sp.vdim[1]) {
      out2.write(r, uint2(gid.x, y), gid.z);
    } else {
#if N >= 3
      y -= sp.vdim[1];
      if (y < sp.vdim[2]) {
        out3.write(r, uint2(gid.x, y), gid.z);
      } else {
#if N >= 4
        y -= sp.vdim[2];
        if (y < sp.vdim[3]) {
          out4.write(r, uint2(gid.x, y), gid.z);
        }
#endif
      }
#endif
    }
  }
#elif V == VX
  int x = gid.x;
  if (x < sp.vdim[0]) {
    out1.write(r, gid.xy, gid.z);
  } else {
    x -= sp.vdim[0];
    if (x < sp.vdim[1]) {
      out2.write(r, uint2(x, gid.y), gid.z);
    } else {
#if N >= 3
      x -= sp.vdim[1];
      if (x < sp.vdim[2]) {
        out3.write(r, uint2(x, gid.y), gid.z);
      } else {
#if N >= 4
        x -= sp.vdim[2];
        if (x < sp.vdim[3]) {
          out4.write(r, uint2(x, gid.y), gid.z);
        }
#endif
      }
#endif
    }
  }
#else
#endif
}

#undef VV
#endif
