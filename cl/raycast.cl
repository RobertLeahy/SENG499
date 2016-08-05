#define KINECT_MAX_DIST (8.0f)
#define KINECT_MIN_DIST (0.4f)
#define STEP_SIZE (0.0005f) // mu *0.8


float getTsdfValue (const int3 vox, const __global float * tsdf, const size_t size) {

    float val = tsdf[vox.x + vox.y * size + vox.z * size * size];

    return val;

}


int3 getVoxel (const float3 pos, const float extent, const size_t size) {

    float flt_size = size;
    float one_over_voxel_size = flt_size / extent;
    return (int3)(
        floor(pos.x * one_over_voxel_size),
        floor(pos.y * one_over_voxel_size),
        floor(pos.z * one_over_voxel_size)
    );

}


int isVoxelValidAndOffBorder (const int3 vox, const size_t size, const size_t dist) {

    if (size < dist) return false;
    int upper = size - dist;
    int lower = dist;
    return !(
        (vox.x < lower) || (vox.x >= upper) ||
        (vox.y < lower) || (vox.y >= upper) ||
        (vox.z < lower) || (vox.z >= upper)
    );

}


int isVoxelValid (const int3 vox, const size_t size) {

    return isVoxelValidAndOffBorder(vox, size, 0);

}


float triLerp (const float3 p, const __global float * tsdf, const float extent, const size_t size) {

    int3 vox = getVoxel(p, extent, size);
    if (!isVoxelValidAndOffBorder(vox,size,1)) return NAN;

    float flt_size = size;
    float voxel_size = extent / flt_size;
    float3 vox_flt = (float3)(vox.x, vox.y, vox.z);
    float3 vox_world = (vox_flt + 0.5f) * voxel_size;

    if (p.x < vox_world.x) --vox.x;
    if (p.y < vox_world.y) --vox.y;
    if (p.z < vox_world.z) --vox.z;

    float3 rs = (p - vox_world) / voxel_size;

    int3 v000 = (int3)(vox.x, vox.y, vox.z);
    int3 v001 = (int3)(vox.x, vox.y, vox.z + 1);
    int3 v010 = (int3)(vox.x, vox.y + 1, vox.z);
    int3 v011 = (int3)(vox.x, vox.y + 1, vox.z + 1);
    int3 v100 = (int3)(vox.x + 1, vox.y, vox.z);
    int3 v101 = (int3)(vox.x + 1, vox.y, vox.z + 1);
    int3 v110 = (int3)(vox.x + 1, vox.y + 1, vox.z);
    int3 v111 = (int3)(vox.x + 1, vox.y + 1, vox.z + 1);

    return
        getTsdfValue(v000, tsdf, size) * (1 - rs.x) * (1 - rs.y) * (1 - rs.z) +
        getTsdfValue(v001, tsdf, size) * (1 - rs.x) * (1 - rs.y) * rs.z +
        getTsdfValue(v010, tsdf, size) * (1 - rs.x) * rs.y * (1 - rs.z) +
        getTsdfValue(v011, tsdf, size) * (1 - rs.x) * rs.y * rs.z +
        getTsdfValue(v100, tsdf, size) * rs.x * (1 - rs.y) * (1 - rs.z) +
        getTsdfValue(v101, tsdf, size) * rs.x * (1 - rs.y) * rs.z +
        getTsdfValue(v110, tsdf, size) * rs.x * rs.y * (1 - rs.z) +
        getTsdfValue(v111, tsdf, size) * rs.x * rs.y * rs.z;

}


/**
 *	Params:
 *
 *  tsdf - TSDF indexed in the normal way
 *  map - Map output
 *  T_g_k - The T_g_k matrix
 *  Kinv - The inverse K matrix
 *  mu - The TSDF truncation distance
 *  extent - The extent, in meters, of the TSDF (assume square volume here)
 *  tsdf_size - The number of elements in a dimension of the TSDF (assume square volume here)
 *  frame_width - The width of the depth frame
 */
 kernel void raycast(
    const __global float * tsdf,    //  0
    __global float * map,   //  1
    const __global float * T_g_k,   //  2
    const __global float * Kinv,    //  3
    const float mu, //  4
    const float extent, //  5
    const unsigned int tsdf_size    //  6
 ) {

	size_t u = get_global_id(0);
    size_t frame_width = get_global_size(0);
	size_t v = get_global_id(1);
    size_t idx = (v * frame_width) + u;
    idx *= 2U;

    //  This is where the camera is in world space
    float3 camera_pos = (float3)(T_g_k[3], T_g_k[7], T_g_k[11]);

    //  We now compute where the pixel is in NDC
    float4 uv_sensor;
    uv_sensor.x = Kinv[0]*u + Kinv[1]*v  + Kinv[2];
    uv_sensor.y = Kinv[3]*u + Kinv[4]*v  + Kinv[5];
    uv_sensor.z = Kinv[6]*u + Kinv[7]*v  + Kinv[8];
    uv_sensor.w = 1.0f;

    //  We now compute a point which relative to the camera's
    //  position in world space gives us the direction along
    //  which we must trace to obtain the value for this pixel 
    float3 uv_world;
    uv_world.x = T_g_k[0]*uv_sensor.x + T_g_k[1]*uv_sensor.y + T_g_k[2]*uv_sensor.z + T_g_k[3]*uv_sensor.w;
    uv_world.y = T_g_k[4]*uv_sensor.x + T_g_k[5]*uv_sensor.y + T_g_k[6]*uv_sensor.z + T_g_k[7]*uv_sensor.w;
    uv_world.z = T_g_k[8]*uv_sensor.x + T_g_k[9]*uv_sensor.y + T_g_k[10]*uv_sensor.z + T_g_k[11]*uv_sensor.w;

    //  Obtain the direction and normalize it
    //
    //  TODO: If this is the null vector what happens?  Can
    //  that happen?  Do we need to check?  How would we
    //  handle that?
    float3 ray_dir = normalize(uv_world - camera_pos);

    //  This gives us the position from which we will begin
    //  our search (i.e. this position is on the near plane)
    float3 initial_ray = camera_pos + KINECT_MIN_DIST * ray_dir;

    //  We trace the ray by starting on the near plane (see
    //  above) and stepping in STEP_SIZE increments until
    //  we find a surface intersection or until we exit the
    //  TSDF volume
    int3 vox = getVoxel(initial_ray, extent, tsdf_size);
    if (!isVoxelValid(vox, tsdf_size)) {

        vstore3(NAN, idx, map);
        vstore3(NAN, idx + 1U, map);

        return;

    }
    float tsdf_val = getTsdfValue(vox, tsdf, tsdf_size);
    float flt_size = tsdf_size;
    float cell_size = extent / flt_size;
    //  We have to start STEP_SIZE away because we need
    //  two samples to detect a sign change
    for (float dist = STEP_SIZE; dist < KINECT_MAX_DIST; dist += STEP_SIZE) {

        float3 where = initial_ray + (ray_dir * dist);

        //  Get current TSDF value
        float tsdf_val_prev = tsdf_val;
        vox = getVoxel(where, extent, tsdf_size);
        if (!isVoxelValid(vox, tsdf_size)) break;
        tsdf_val = getTsdfValue(vox, tsdf, tsdf_size);

        if (isnan(tsdf_val)) continue;
        if (isnan(tsdf_val_prev)) continue;

        int p = signbit(tsdf_val_prev);
        int c = signbit(tsdf_val);
        if (p == c) continue;

        //  Detect backface: From negative to positive
        if (p) break;

        //  Good sign change

        float ftdt = triLerp(where, tsdf, extent, tsdf_size);
        if (isnan(ftdt)) break;

        float3 last = where - (ray_dir * STEP_SIZE);
        float ft = triLerp(last, tsdf, extent, tsdf_size);
        if (isnan(ft)) break;

        float t_star = dist - (STEP_SIZE * ft) / (ftdt - ft);

        //  Store computed vertex position
        float3 v = initial_ray + (ray_dir * t_star);
        vstore3(v, idx, map);

        //  Check to see if we can even compute a normal
        if (!isVoxelValidAndOffBorder(getVoxel(last, extent, tsdf_size), tsdf_size, 2)) {

            vstore3(NAN, idx + 1U, map);
            return;

        }

        //  Compute a normal by computing partial derivatives
        //  along all three axes
        float3 t = v;
        t.x += cell_size;
        float fx1 = triLerp(t, tsdf, extent, tsdf_size);
        t = v;
        t.x -= cell_size;
        float fx2 = triLerp(t, tsdf, extent, tsdf_size);

        t = v;
        t.y += cell_size;
        float fy1 = triLerp(t, tsdf, extent, tsdf_size);
        t = v;
        t.y -= cell_size;
        float fy2 = triLerp(t, tsdf, extent, tsdf_size);

        t = v;
        t.z += cell_size;
        float fz1 = triLerp(t, tsdf, extent, tsdf_size);
        t = v;
        t.z -= cell_size;
        float fz2 = triLerp(t, tsdf, extent, tsdf_size);

        //  We only store the normal if it's not the null vector
        //  otherwise we store NaN
        float3 n = (float3) (fx1 - fx2, fy1 - fy2, fz1 - fz2);
        float3 nullv = (float3)(0, 0, 0);
        vstore3((n == nullv) ? NAN : normalize(n), idx + 1U, map);

        return;

    }

    vstore3(NAN, idx, map);
    vstore3(NAN, idx + 1U, map);

}

