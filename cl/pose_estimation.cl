float3 matrixmul3(const __global float * m, float3 v) {

    float3 retr;
    retr.x=dot(vload3(0,m),v);
    retr.y=dot(vload3(1,m),v);
    retr.z=dot(vload3(2,m),v);

    return retr;

}


float4 matrixmul4(const __global float * m, float4 v) {

    float4 retr;
    retr.x=dot(vload4(0,m),v);
    retr.y=dot(vload4(1,m),v);
    retr.z=dot(vload4(2,m),v);
    retr.w=dot(vload4(3,m),v);

    return retr;

}


int is_finite(float3 v) {

	return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);

}


void zero_ab (__global float * a, __global float * b) {

	#pragma unroll
	for (size_t i=0;i<(6*6);++i) a[i]=0;
	#pragma unroll
	for (size_t i=0;i<6;++i) b[i]=0;

}


void icp (float3 p, float3 q, float3 n, __global float * a, __global float * b) {

	float3 c=cross(p,n);
	float pqn=dot(p-q,n);

	//	First row
	a[0]=c.x*c.x;
	a[1]=c.x*c.y;
	a[2]=c.x*c.z;
	a[3]=c.x*n.x;
	a[4]=c.x*n.y;
	a[5]=c.x*n.z;
	//	Second row
	a[6]=c.y*c.y;
	a[7]=c.y*c.z;
	a[8]=c.y*n.x;
	a[9]=c.y*n.y;
	a[10]=c.y*n.z;
	//	Third row
	a[11]=c.z*c.z;
	a[12]=c.z*n.x;
	a[13]=c.z*n.y;
	a[14]=c.z*n.z;
	//	Fourth row
	a[15]=n.x*n.x;
	a[16]=n.x*n.y;
	a[17]=n.x*n.z;
	//	Fifth row
	a[18]=n.y*n.y;
	a[19]=n.y*n.z;
	//	Sixth row
	a[20]=n.z*n.z;

	pqn*=-1;
	b[0]=c.x*pqn;
	b[1]=c.y*pqn;
	b[2]=c.z*pqn;
	b[3]=n.x*pqn;
	b[4]=n.y*pqn;
	b[5]=n.z*pqn;

}


struct __attribute__((packed)) mats {

	float a [21U];	//	The matrix is symmetric so we don't store the lower triangle
	float b [6U];

};


kernel void correspondences(
	const __global float * map,	//	0
	const __global float * prev_map,	//	1
	const __global float * t_gk_prev_inverse,	//	2
	const __global float * t_z,	//	3
	float epsilon_d,	//	4
	float epsilon_theta,	//	5
	const __global float * k,	//	6
	__global struct mats * mats
) {

	size_t x=get_global_id(0);
	size_t width=get_global_size(0);
	size_t y=get_global_id(1);
	size_t idx=(y*width)+x;
	global struct mats * ms=mats+idx;
	idx*=2U;

	float3 curr_pv=vload3(idx,prev_map);
	if (!is_finite(curr_pv)) {

		zero_ab(ms->a,ms->b);
		return;

	}

	float4 curr_pv_homo=(float4)(curr_pv,1);

	float4 v_pcp_h=matrixmul4(t_gk_prev_inverse,curr_pv_homo);

	float3 v_pcp=(float3)(v_pcp_h.x, v_pcp_h.y, v_pcp_h.z);
	float3 uv3=matrixmul3(k,v_pcp);

	int2 u=(int2)(round(uv3.x/uv3.z),round(uv3.y/uv3.z));

	if ((u.x<0) || (u.y<0) || (((size_t)u.x)>=width) || (((size_t)u.y)>=get_global_size(1))) {

		zero_ab(ms->a,ms->b);
		return;

	}

	size_t lin_idx=u.x+u.y*width;

	float3 curr_pn=vload3(idx+1U,prev_map);
	//	These are in current camera space
	lin_idx*=2U;
	float3 curr_v=vload3(lin_idx,map);
	float3 curr_n=vload3(lin_idx+1U,map);
	//	TODO: Should we be checking the current normal?
	if (!(is_finite(curr_v) && is_finite(curr_pn))) {

		zero_ab(ms->a,ms->b);
		return;

	}

	float4 curr_v_homo=(float4)(curr_v,1);
	float4 t_z_curr_v_homo=matrixmul4(t_z,curr_v_homo);
	float4 t_z_curr_v_homo_curr_pv_homo=t_z_curr_v_homo-curr_pv_homo;
	float3 t_z_curr_v=(float3)(t_z_curr_v_homo.x,t_z_curr_v_homo.y,t_z_curr_v_homo.z);
	if (dot(t_z_curr_v_homo_curr_pv_homo,t_z_curr_v_homo_curr_pv_homo)>(epsilon_d*epsilon_d)) {

		zero_ab(ms->a,ms->b);
		return;

	}

	float3 r_z_curr_n;
	r_z_curr_n.x=
		(t_z[0]*curr_n.x)+
		(t_z[1]*curr_n.y)+
		(t_z[2]*curr_n.z);
	r_z_curr_n.y=
		(t_z[4]*curr_n.x)+
		(t_z[5]*curr_n.y)+
		(t_z[6]*curr_n.z);
	r_z_curr_n.z=
		(t_z[8]*curr_n.x)+
		(t_z[9]*curr_n.y)+
		(t_z[10]*curr_n.z);
	float3 crzcncpn=cross(r_z_curr_n,curr_pn);
	if (dot(crzcncpn,crzcncpn)>(epsilon_theta*epsilon_theta)) {

		zero_ab(ms->a,ms->b);
		return;

	}

	icp(t_z_curr_v,curr_pv,curr_pn,ms->a,ms->b);

}


void matrixadd6 (__global float * restrict a, const __global float * restrict b) {

	#pragma unroll
	for (size_t i=0;i<21U;++i) a[i]+=b[i];

}


void vectoradd6 (__global float * restrict a, const __global float * restrict b) {

	#pragma unroll
	for (size_t i=0;i<6U;++i) a[i]+=b[i];

}


//	ASSUMPTION:
//
//	Local group size IS NOT ODD!!!!
kernel void parallel_sum(
	const __global struct mats * src,	//	0
	__global struct mats * dest,	//	1
	__local struct mats * lmats	//	2
) {

	//	Get identifiers
	size_t g=get_global_id(0);
	size_t l=get_local_id(0);

	//	Load into local memory
	lmats[l]=src[g];

	for (size_t stride=get_local_size(0)/2U;stride>0;stride/=2U) {

		//	Wait for everything else to finish
		barrier(CLK_LOCAL_MEM_FENCE);

		//	Perform add if needed
		if (l>=stride) continue;

		local float * a=lmats[l].a;
		const local float * b=lmats[l+stride].a;
		#pragma unroll
		for (size_t i=0;i<27U;++i) a[i]+=b[i]; 

	}

	//	Write out
	if (l==0) dest[get_group_id(0)]=lmats[0];

}


kernel void serial_sum(
	__global struct mats * mats,	//	0
	unsigned int length	//	1
) {

	size_t i=get_global_id(0);
	float sum=0;

	for (size_t j=0;j<length;++j) sum+=mats[j].a[i];

	mats[0].a[i]=sum;

}
