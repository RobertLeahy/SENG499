#define SMALL_DEPTH (0.1f)

kernel void vertex_map(const __global float * src, __global float * dest, const __global float * K_inv) {

    size_t x = get_global_id(0);
    size_t width = get_global_size(0);
    size_t y = get_global_id(1);
    size_t idx = (y * width) + x;
    float depth = src[idx];

    float3 v;
    if (isnan(depth) || (depth <= SMALL_DEPTH)) {

        v = NAN;

    } else {

        // assume z = 1 -> homoegeneous coords
        float v1 = K_inv[0]*x + K_inv[1]*y + K_inv[2];
        float v2 = K_inv[3]*x + K_inv[4]*y + K_inv[5];
        float v3 = K_inv[6]*x + K_inv[7]*y + K_inv[8];

        v = depth * (float3)(v1, v2, v3);

    }

    vstore3(v, idx * 2U, dest);

}