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


kernel void correspondences(
	const __global float * v,	//	0
	const __global float * n,	//	1
	const __global float * pv,	//	2
	const __global float * pn,	//	3
	const unsigned int width,	//	4
	const unsigned int height,	//	5
	const __global float * t_frame_frame,	//	6
	const __global float * t_z,	//	7
	float epsilon_d,	//	8
	float epsilon_theta,	//	9
	const __global float * k,	//	10
	__global float * corr_pv,	//	11
	__global float * corr_pn,	//	12
	volatile __global unsigned int * count	//	13
) {

	size_t x=get_global_id(0);
	size_t y=get_global_id(1);
	size_t idx=(y*width)+x;

	float3 curr_v=vload3(idx,v);
	float4 curr_v_homo=(float4)(curr_v,1);
	
	float4 v_cp_h=matrixmul4(t_frame_frame,curr_v_homo);
	
	float3 v_cp=(float3)(v_cp_h.x, v_cp_h.y, v_cp_h.z);
	float3 uv3=matrixmul3(k,v_cp);

	int2 u=(int2)(round(uv3.x/uv3.z),round(uv3.y/uv3.z));

	int lin_idx=u.x+u.y*width;

	float3 nullv=(float3)(0,0,0);
	if (lin_idx>=(width*height) || lin_idx<0) {

		vstore3(nullv,idx,corr_pn);
		atomic_add(count,1);
		return;

	}

	float3 curr_n=vload3(idx,n);
	float3 curr_pv=vload3(lin_idx,pv);
	float3 curr_pn=vload3(lin_idx,pn);
	//	TODO: Should we be checking the current normal?
	if (!(is_finite(curr_v) && is_finite(curr_pv) && is_finite(curr_pn))) {

		vstore3(nullv,idx,corr_pn);
		atomic_add(count,1);
		return;

	}

	float4 curr_pv_homo=(float4)(curr_pv,1);
	float4 t_z_curr_v_homo=matrixmul4(t_z,curr_v_homo);
	float4 t_z_curr_v_homo_curr_pv_homo=t_z_curr_v_homo-curr_pv_homo;
	if (dot(t_z_curr_v_homo_curr_pv_homo,t_z_curr_v_homo_curr_pv_homo)>(epsilon_d*epsilon_d)) {

		vstore3(nullv,idx,corr_pn);
		atomic_add(count,1);
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
	float3 c=cross(r_z_curr_n,curr_pn);
	if (dot(c,c)>(epsilon_theta*epsilon_theta)) {

		vstore3(nullv,idx,corr_pn);
		atomic_add(count,1);
		return;

	}

	vstore3(curr_pv,idx,corr_pv);
	vstore3(curr_pn,idx,corr_pn);

}


kernel void map(
	const __global float * v,	//	0
	const __global float * corr_pv,	//	1
	const __global float * corr_pn,	//	2
	const unsigned int width,	//	3
	const unsigned int height,	//	4
	__global float * ais,	//	5
	__global float * bis	//	6
) {

	size_t x=get_global_id(0);
	size_t y=get_global_id(1);
	size_t idx=(y*width)+x;
	size_t bi_idx=idx*6;
	global float * bi=bis+bi_idx;
	size_t ai_idx=bi_idx*6;
	global float * ai=ais+ai_idx;


	//	We check to make sure that this was a valid
	//	correspondence, if not we count it and
	//	bail out
	//
	//	Counting invalid correspondences suggested
	//	by Jordan as a way to minimize atomic adds
	float3 n=vload3(idx,corr_pn);
	if ((n.x==0) && (n.y==0) && (n.z==0)) {

		#pragma unroll
		for (size_t i=0;i<(6*6);++i) ai[i]=0;
		#pragma unroll
		for (size_t i=0;i<6;++i) bi[i]=0;

		return;

	}

	float3 p=vload3(idx,v);
	float3 q=vload3(idx,corr_pv);

	float3 c=cross(p,n);
	float pqn=dot(p-q,n);

	//	First row
	ai[0]=c.x*c.x;
	ai[1]=c.x*c.y;
	ai[2]=c.x*c.z;
	ai[3]=c.x*n.x;
	ai[4]=c.x*n.y;
	ai[5]=c.x*n.z;
	//	Second row
	ai[6]=c.y*c.x;
	ai[7]=c.y*c.y;
	ai[8]=c.y*c.z;
	ai[9]=c.y*n.x;
	ai[10]=c.y*n.y;
	ai[11]=c.y*n.z;
	//	Third row
	ai[12]=c.z*c.x;
	ai[13]=c.z*c.y;
	ai[14]=c.z*c.z;
	ai[15]=c.z*n.x;
	ai[16]=c.z*n.y;
	ai[17]=c.z*n.z;
	//	Fourth row
	ai[18]=n.x*c.x;
	ai[19]=n.x*c.y;
	ai[20]=n.x*c.z;
	ai[21]=n.x*n.x;
	ai[22]=n.x*n.y;
	ai[23]=n.x*n.z;
	//	Fifth row
	ai[24]=n.y*c.x;
	ai[25]=n.y*c.y;
	ai[26]=n.y*c.z;
	ai[27]=n.y*n.x;
	ai[28]=n.y*n.y;
	ai[29]=n.y*n.z;
	//	Sixth row
	ai[30]=n.z*c.x;
	ai[31]=n.z*c.y;
	ai[32]=n.z*c.z;
	ai[33]=n.z*n.x;
	ai[34]=n.z*n.y;
	ai[35]=n.z*n.z;

	pqn*=-1;
	bi[0]=c.x*pqn;
	bi[1]=c.y*pqn;
	bi[2]=c.z*pqn;
	bi[3]=n.x*pqn;
	bi[4]=n.y*pqn;
	bi[5]=n.z*pqn;

}


kernel void reduce_a(
	const __global float * ais,	//	0
	__global float * a,	//	1
	const unsigned int length	//	2
) {

	size_t x=get_global_id(0);
	size_t y=get_global_id(1);
	size_t idx=(y*6)+x;

	float sum=0;
	for (size_t i=0;i<length;++i) {

		sum+=ais[idx];
		ais+=6*6;

	}
	a[idx]=sum;

}


kernel void reduce_b(
	const __global float * bis,	//	0
	__global float * b,	//	1
	const unsigned int length	//	2
) {

	size_t idx=get_global_id(0);

	float sum=0;
	for (size_t i=0;i<length;++i) {

		sum+=bis[idx];
		bis+=6;

	}
	b[idx]=sum;

}
