#include <dynfu/kinect_fusion.hpp>
#include <Eigen/Dense>
#include <stdexcept>
#include <utility>


namespace dynfu {


	void kinect_fusion::check_pipeline () const {

		if (!(dd_ && mpb_ && pepb_ && urpb_ && sppb_)) throw std::logic_error(
			"Incomplete Kinect Fusion pipeline"
		);

	}
	
	
	void kinect_fusion::get_frame () {
		
		dynfu::depth_device::value_type f;
		using std::swap;
		swap(f,frame_);
		timer t;
		frame_=(*dd_)(std::move(f));
		ddt_=t.elapsed();
		
	}
	
	
	void kinect_fusion::get_vertex_and_normal_map () {
		
		dynfu::measurement_pipeline_block::value_type map;
		using std::swap;
		swap(map,map_);
		timer t;
		map_=(*mpb_)(*frame_,width_,height_,k_,std::move(map));
		mpbt_=t.elapsed();
		
	}


	void kinect_fusion::get_tgk () {

		dynfu::pose_estimation_pipeline_block::value_type t_g_k;
		using std::swap;
		swap(t_g_k,t_g_k_);
		timer t;
		t_g_k_=(*pepb_)(*map_,prev_map_.get(),k_,std::move(t_g_k));
		pepbt_=t.elapsed();

	}


	void kinect_fusion::get_tsdf () {

		dynfu::update_reconstruction_pipeline_block::value_type tsdf{{},0,0,0};
		using std::swap;
		swap(tsdf,tsdf_);
		timer t;
		tsdf_=(*urpb_)(*frame_,width_,height_,k_,*t_g_k_,std::move(tsdf));
		urpbt_=t.elapsed();

	}


	void kinect_fusion::get_prediction () {

		dynfu::surface_prediction_pipeline_block::value_type map;
		using std::swap;
		swap(map,prev_map_);
		timer t;
		prev_map_=(*sppb_)(*tsdf_.buffer,tsdf_.width,tsdf_.height,tsdf_.depth,*t_g_k_,k_,*map_,std::move(map));
		sppbt_=t.elapsed();

	}
	
	
	kinect_fusion::kinect_fusion () noexcept
		:	dd_(nullptr),
			k_(Eigen::Matrix3f::Zero()),
			width_(0),
			height_(0),
			mpb_(nullptr),
			pepb_(nullptr),
			urpb_(nullptr),
			sppb_(nullptr),
			tsdf_{{},0,0,0}
	{	}
	
	
	void kinect_fusion::depth_device (dynfu::depth_device & dd) noexcept {
		
		dd_=&dd;
		k_=dd.k();
		width_=dd.width();
		height_=dd.height();
		
	}
	
	
	void kinect_fusion::measurement_pipeline_block (dynfu::measurement_pipeline_block & mpb) noexcept {
		
		mpb_=&mpb;
		
	}


	void kinect_fusion::pose_estimation_pipeline_block (dynfu::pose_estimation_pipeline_block & pepb) noexcept {

		pepb_=&pepb;

	}


	void kinect_fusion::update_reconstruction_pipeline_block (dynfu::update_reconstruction_pipeline_block & urpb) noexcept {

		urpb_=&urpb;

	}


	void kinect_fusion::surface_prediction_pipeline_block (dynfu::surface_prediction_pipeline_block & sppb) noexcept {

		sppb_=&sppb;

	}
	
	
	void kinect_fusion::operator () () {

		check_pipeline();
		
		get_frame();
		get_vertex_and_normal_map();
		get_tgk();
		get_tsdf();
		get_prediction();
		
	}
	
	
	kinect_fusion::timer::duration kinect_fusion::depth_device_elapsed () const {
		
		return ddt_;
		
	}
	
	
	kinect_fusion::timer::duration kinect_fusion::measurement_pipeline_block_elapsed () const {
		
		return mpbt_;
		
	}


	kinect_fusion::timer::duration kinect_fusion::pose_estimation_pipeline_block_elapsed () const {

		return pepbt_;

	}


	kinect_fusion::timer::duration kinect_fusion::update_reconstruction_pipeline_block_elapsed () const {

		return urpbt_;

	}


	kinect_fusion::timer::duration kinect_fusion::surface_prediction_pipeline_block_elapsed () const {

		return sppbt_;

	}


	depth_device::value_type::element_type & kinect_fusion::frame () const noexcept {

		return *frame_;

	}


	measurement_pipeline_block::value_type::element_type & kinect_fusion::vertex_and_normal_map () const noexcept {

		return *map_;

	}


	pose_estimation_pipeline_block::value_type::element_type & kinect_fusion::pose_estimation () const noexcept {

		return *t_g_k_;

	}


	update_reconstruction_pipeline_block::value_type::element_type & kinect_fusion::truncated_signed_distance_function () const noexcept {

		return *tsdf_.buffer;

	}


	std::size_t kinect_fusion::truncated_signed_distance_function_width () const noexcept {

		return tsdf_.width;

	}


	std::size_t kinect_fusion::truncated_signed_distance_function_height () const noexcept {

		return tsdf_.height;

	}


	std::size_t kinect_fusion::truncated_signed_distance_function_depth () const noexcept {

		return tsdf_.depth;

	}


	surface_prediction_pipeline_block::value_type::element_type & kinect_fusion::predicted_vertex_and_normal_map () const noexcept {

		return *prev_map_;

	}
	
	
}
