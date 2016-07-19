#include <dynfu/boost_compute_detail_type_name_trait.hpp>
#include <dynfu/kinect_fusion_opencl_surface_prediction_pipeline_block.hpp>
#include <dynfu/opencl_vector_pipeline_value.hpp>
#include <Eigen/Dense>
#include <cstddef>
#include <memory>
#include <utility>


namespace dynfu {
	
	
	static boost::compute::kernel get_raycast_kernel (opencl_program_factory & opf) {
		
		auto p=opf("raycast");
		return boost::compute::kernel(p,"raycast");
		
	}	
	
	
	kinect_fusion_opencl_surface_prediction_pipeline_block::kinect_fusion_opencl_surface_prediction_pipeline_block (
		boost::compute::command_queue q,
		opencl_program_factory & opf,
		float mu,
		std::size_t tsdf_size,
		float tsdf_extent,
		std::size_t frame_width,
		std::size_t frame_height)
		:	ve_(std::move(q)),
			raycast_kernel_(get_raycast_kernel(opf)),
			t_g_k_buf_(ve_.command_queue().get_context(),sizeof(Eigen::Matrix4f),CL_MEM_READ_ONLY),
			ik_buf_(ve_.command_queue().get_context(),sizeof(Eigen::Matrix3f),CL_MEM_READ_ONLY),
			mu_(mu),
			tsdf_extent_(tsdf_extent),
			tsdf_size_(tsdf_size),
			frame_width_(frame_width),
			frame_height_(frame_height)
	{

		raycast_kernel_.set_arg(8, std::uint32_t(frame_width_));
		raycast_kernel_.set_arg(7, std::uint32_t(tsdf_size_));
		raycast_kernel_.set_arg(6, tsdf_extent_);
		raycast_kernel_.set_arg(5, mu_);
		raycast_kernel_.set_arg(4, ik_buf_);
		raycast_kernel_.set_arg(3, t_g_k_buf_);

	}
	
	
	kinect_fusion_opencl_surface_prediction_pipeline_block::value_type kinect_fusion_opencl_surface_prediction_pipeline_block::operator () (
		update_reconstruction_pipeline_block::value_type tsdf,
		Eigen::Matrix3f k,
		Eigen::Matrix4f t_g_k,
		kinect_fusion_opencl_surface_prediction_pipeline_block::value_type v
	) {

		auto q = ve_.command_queue();


		if (t_g_k_ != t_g_k) {
			
			t_g_k_ = t_g_k;
			t_g_k.transposeInPlace();
			q.enqueue_write_buffer(t_g_k_buf_, 0, sizeof(t_g_k), t_g_k.data());
		
		}


		if (k_ != k) {

			k_ = k;

			Eigen::Matrix3f k_inv = k.inverse();
			k_inv.transposeInPlace();
			q.enqueue_write_buffer(ik_buf_,0,sizeof(k_inv),k_inv.data());

		}

		auto num_depth_px = frame_width_ * frame_height_;


		// allocate space if needed for the vmap and nmap
		auto && vmap_ptr = std::get<0>(v);
		auto && nmap_ptr = std::get<1>(v);
		using type=opencl_vector_pipeline_value<Eigen::Vector3f>;
		if (!vmap_ptr) vmap_ptr=std::make_unique<type>(q);
		if (!nmap_ptr) nmap_ptr=std::make_unique<type>(q);
		auto && vmap = dynamic_cast<type &>(*vmap_ptr).vector();
		auto && nmap = dynamic_cast<type &>(*nmap_ptr).vector();
		vmap.resize(num_depth_px, q);
		nmap.resize(num_depth_px, q);


		auto && tsdf_ptr = tsdf.buffer;
		using tsdf_type = opencl_vector_pipeline_value<float>;
		if (!tsdf_ptr) tsdf_ptr = std::make_unique<tsdf_type>(q);
		auto && tsdf_buf = dynamic_cast<tsdf_type &>(*tsdf_ptr).vector();
		//tsdf_buf.resize(tsdf_width_*tsdf_height_*tsdf_depth_, q);

		raycast_kernel_.set_arg(0,tsdf_buf);
		raycast_kernel_.set_arg(1,vmap);
		raycast_kernel_.set_arg(2,nmap);

		std::size_t extent []={frame_width_,frame_height_};
		q.enqueue_nd_range_kernel(raycast_kernel_,2,nullptr,extent,nullptr);

		return v;
	}
	
}
