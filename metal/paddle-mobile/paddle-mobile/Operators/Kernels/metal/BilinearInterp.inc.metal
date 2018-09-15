#ifdef P

#define CONCAT2(a, b) a ## b
#define CONCAT2_(a, b) a ## _ ## b

#define FUNC(f, p) CONCAT2_(f, p)
#define VECTOR(p, n) CONCAT2(p, n)

kernel void FUNC(bilinear_interp, P)(texture2d_array<P, access::read> input [[texture(0)]],
                     texture2d_array<P, access::write> output [[texture(1)]],
                     constant bilinear_interp_param & pm [[buffer(0)]],
                     uint3 gid [[thread_position_in_grid]]) {
  VECTOR(P, 4) r;
  if ((input.get_width() == output.get_width()) && (input.get_height() == output.get_height())) {
    r = input.read(gid.xy, gid.z);
  } else {
    P w = gid.x * pm.ratio_w;
    P h = gid.y * pm.ratio_h;
    uint w0 = w, h0 = h;
    uint w1 = w0 + 1, h1 = h0 + 1;
    P w1lambda = w - w0, h1lambda = h - h0;
    P w2lambda = 1.0 - w1lambda, h2lambda = 1.0 - h1lambda;
    if (w1 >= input.get_width()) w1 = w0;
    if (h1 >= input.get_height()) h1 = h0;
    VECTOR(P, 4) r0 = input.read(uint2(w0, h0), gid.z);
    VECTOR(P, 4) r1 = input.read(uint2(w1, h0), gid.z);
    VECTOR(P, 4) r2 = input.read(uint2(w0, h1), gid.z);
    VECTOR(P, 4) r3 = input.read(uint2(w1, h1), gid.z);
    r = h2lambda * (w2lambda * r0 + w1lambda * r1)
      + h1lambda * (w2lambda * r2 + w1lambda * r3);
  }
  output.write(r, gid.xy, gid.z);
}

#endif
