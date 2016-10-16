#include <boost/compute.hpp>
#include <kinfu/boost_compute_detail_type_name_trait.hpp>
#include <kinfu/cpu_pipeline_value.hpp>
#include <kinfu/kinect_fusion_opencl_pose_estimation_pipeline_block.hpp>
#include <kinfu/opencl_program_factory.hpp>
#include <kinfu/optional.hpp>
#include <kinfu/scope.hpp>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>


namespace kinfu {


	constexpr std::size_t a_floats=21U;
	constexpr std::size_t b_floats=6U;
	constexpr std::size_t mats_floats=a_floats+b_floats;
	constexpr std::size_t sizeof_mats=sizeof(float)*mats_floats;


	kinect_fusion_opencl_pose_estimation_pipeline_block::kinect_fusion_opencl_pose_estimation_pipeline_block (
		boost::compute::command_queue q,
		opencl_program_factory & pf,
		float epsilon_d,
		float epsilon_theta,
		std::size_t frame_width,
		std::size_t frame_height,
		Eigen::Matrix4f t_gk_initial,
		std::size_t numit,
		std::size_t group_size,
		bool force_px_px
	)	:	q_(std::move(q)),
			t_z_(q_.get_context(),sizeof(Eigen::Matrix4f),CL_MEM_READ_ONLY|CL_MEM_HOST_WRITE_ONLY),
			t_frame_frame_(q_.get_context(),sizeof(Eigen::Matrix4f),CL_MEM_READ_ONLY|CL_MEM_HOST_WRITE_ONLY),
			k_(q_.get_context(),sizeof(Eigen::Matrix3f),CL_MEM_READ_ONLY|CL_MEM_HOST_WRITE_ONLY),
			e_(q_),
			p_e_(q_),
			epsilon_d_(epsilon_d),
			epsilon_theta_(epsilon_theta),
			frame_width_(frame_width),
			frame_height_(frame_height),
			t_gk_initial_(std::move(t_gk_initial)),
			numit_(numit),
			group_size_(group_size),
			force_px_px_(force_px_px)
	{

		if (numit_==0) throw std::logic_error("Must iterate at least once");

		if (group_size_==0) throw std::logic_error("Size of OpenCL parallel sum work groups must be at least 1");
		auto frame_size=frame_width_*frame_height_;
		if (group_size_>frame_size) {

			std::ostringstream ss;
			ss << "Size of OpenCL parallel sum work groups " << group_size_ << " is greater than frame size " << frame_size << " (" << frame_width_ << "x" << frame_height_ << ")";
			throw std::logic_error(ss.str());

		}
		if ((frame_size%group_size_)!=0) {

			std::ostringstream ss;
			ss << "Size of OpenCL parallel sum work groups " << group_size_ << " does not evenly divide frame size " << frame_size << " (" << frame_width_ << "x" << frame_height_ << ")";
			throw std::logic_error(ss.str());

		}

		auto p=pf("pose_estimation");
		corr_=p.create_kernel("correspondences");
		parallel_sum_=p.create_kernel("parallel_sum");
		serial_sum_=p.create_kernel("serial_sum");

		group_size_ = std::min(group_size_, corr_.get_work_group_info<std::size_t>(q_.get_device(), CL_KERNEL_WORK_GROUP_SIZE));

		auto frame_size_after=frame_size/group_size_;
		mats_=boost::compute::buffer(q_.get_context(),frame_size_after*sizeof_mats);
		if (
			(group_size_!=1) &&
			((frame_size_after%group_size_)==0)
		) mats_output_=boost::compute::buffer(q_.get_context(),(frame_size_after/group_size_)*sizeof_mats);


		boost::compute::local_buffer<float> scratch(mats_floats*group_size_);

		//	Correspondences arguments
		corr_.set_arg(2,t_frame_frame_);
		corr_.set_arg(3,t_z_);
		corr_.set_arg(4,epsilon_d_);
		corr_.set_arg(5,epsilon_theta_);
		corr_.set_arg(6,k_);
		corr_.set_arg(7,std::uint32_t(frame_width));
		corr_.set_arg(8,std::uint32_t(frame_height));
		corr_.set_arg(9,mats_);
		corr_.set_arg(10,scratch);

		//	Parallel sum arguments
		parallel_sum_.set_arg(2,scratch);

	}


	kinect_fusion_opencl_pose_estimation_pipeline_block::value_type kinect_fusion_opencl_pose_estimation_pipeline_block::operator () (
		measurement_pipeline_block::value_type::element_type & map,
		measurement_pipeline_block::value_type::element_type * prev_map,
		Eigen::Matrix3f k,
		value_type t_gk_minus_one
	) {

		using pv_type=cpu_pipeline_value<value_type::element_type::type>;
	
		if (!(t_gk_minus_one && prev_map)) {

			t_gk_minus_one=std::make_unique<pv_type>();
			static_cast<pv_type &>(*t_gk_minus_one).emplace(t_gk_initial_);

			return t_gk_minus_one;

		}

		//	Get input vectors and bind parameters
		auto && map_b=e_(map);
		corr_.set_arg(0,map_b);
		auto && prev_map_b=p_e_(*prev_map);
		corr_.set_arg(1,prev_map_b);

		auto && pv=dynamic_cast<pv_type &>(*t_gk_minus_one);
		auto t_gk_minus_one_m=pv.get();
		Eigen::Matrix4f t_z(t_gk_minus_one_m);
		Eigen::Matrix4f t_gk_prev_inverse(t_gk_minus_one_m.inverse());

		k.transposeInPlace();
		auto kw=q_.enqueue_write_buffer_async(k_,0,sizeof(k),&k);
		auto kwg=make_scope_exit([&] () noexcept {	kw.wait();	});

		for (std::size_t i=0;i<numit_;++i) {

			//	Upload current estimate to the GPU
			Eigen::Matrix4f t_frame_frame(t_gk_prev_inverse*t_z);

			// If enabled, force pixel to pixel correspondences
			if (force_px_px_) t_frame_frame = Eigen::Matrix4f::Identity();

			t_z.transposeInPlace();
			auto tzw=q_.enqueue_write_buffer_async(t_z_,0,sizeof(t_z),&t_z);
			auto tzwg=make_scope_exit([&] () noexcept {	tzw.wait();	});
			t_frame_frame.transposeInPlace();
			auto tffw=q_.enqueue_write_buffer_async(t_frame_frame_,0,sizeof(t_frame_frame),&t_frame_frame);
			auto tffwg=make_scope_exit([&] () noexcept {	tffw.wait();	});

			//	Enqueue correspondences kernel
			auto input_size=frame_height_*frame_width_;
			q_.enqueue_nd_range_kernel(corr_,1,nullptr,&input_size,&group_size_);
			input_size/=group_size_;

			//	Perform parallel phase of sum
			boost::compute::buffer in(mats_);
			boost::compute::buffer out(mats_output_);
			std::size_t output_size=input_size;	//	In case the branch on the next line isn't taken
			if (group_size_!=1) while ((input_size%group_size_)==0) {

				output_size=input_size/group_size_;

				parallel_sum_.set_arg(0,in);
				parallel_sum_.set_arg(1,out);
				q_.enqueue_nd_range_kernel(parallel_sum_,1,nullptr,&input_size,&group_size_);

				input_size=output_size;
				using std::swap;
				swap(in,out);

			}

			//	Perform serial phase of sum if necessary
			if (output_size>1) {

				serial_sum_.set_arg(0,in);
				serial_sum_.set_arg(1,std::uint32_t(input_size));
				std::size_t serial_size(27);
				q_.enqueue_nd_range_kernel(serial_sum_,1,nullptr,&serial_size,nullptr);

			}

			//	Collect results
			float buffer [mats_floats];
			q_.enqueue_read_buffer(in,0,sizeof(buffer),buffer);
			using a_type=Eigen::Matrix<float,6,6>;
			a_type a;
			a <<	buffer[0],buffer[1],buffer[2],buffer[3],buffer[4],buffer[5],
					buffer[1],buffer[6],buffer[7],buffer[8],buffer[9],buffer[10],
					buffer[2],buffer[7],buffer[11],buffer[12],buffer[13],buffer[14],
					buffer[3],buffer[8],buffer[12],buffer[15],buffer[16],buffer[17],
					buffer[4],buffer[9],buffer[13],buffer[16],buffer[18],buffer[19],
					buffer[5],buffer[10],buffer[14],buffer[17],buffer[19],buffer[20];
			using b_type=Eigen::Matrix<float,6,1>;
			b_type b;
			std::memcpy(&b,buffer+a_floats,sizeof(b));

			//	Solve system
			auto solver=a.ldlt();
			b_type x=solver.solve(b);

			float alpha=x(0);
			float beta=x(1);
			float gamma=x(2);
			float tx=x(3);
			float ty=x(4);
			float tz=x(5);

			Eigen::Matrix4f t_inc;
			t_inc <<  1.0f, -gamma, beta, tx,
					 gamma, 1.0f, -alpha, ty,
				 	 -beta, alpha, 1.0f, tz,
					 0.0f, 0.0f, 0.0f, 1.0f;
			t_z.transposeInPlace();
			t_z=t_inc*t_z;

		}

		pv.emplace(t_z);

		return t_gk_minus_one;
	
	}


}
