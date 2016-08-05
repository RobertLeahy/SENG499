#include <boost/compute.hpp>
#include <dynfu/boost_compute_detail_type_name_trait.hpp>
#include <dynfu/opencl_vector_pipeline_value.hpp>
#include <dynfu/kinect_fusion_opencl_frame_to_frame_surface_prediction_pipeline_block.hpp>
#include <dynfu/pixel.hpp>
#include <Eigen/Dense>
#include <memory>
#include <tuple>
#include <utility>


namespace dynfu {


	kinect_fusion_opencl_frame_to_frame_surface_prediction_pipeline_block::kinect_fusion_opencl_frame_to_frame_surface_prediction_pipeline_block (boost::compute::command_queue q) : q_(std::move(q)) {	}


	kinect_fusion_opencl_frame_to_frame_surface_prediction_pipeline_block::value_type kinect_fusion_opencl_frame_to_frame_surface_prediction_pipeline_block::operator () (
		update_reconstruction_pipeline_block::value_type::element_type &,
		std::size_t,
		std::size_t,
		std::size_t,
		pose_estimation_pipeline_block::value_type::element_type &,
		Eigen::Matrix3f,
		measurement_pipeline_block::value_type::element_type & prev,
		value_type vn
	) {

		using v_type=opencl_vector_pipeline_value<dynfu::pixel>;
		if (!vn) vn=std::make_unique<v_type>(q_);
		auto && pv=dynamic_cast<v_type &>(*vn);
		auto && buffer=pv.vector();

		opencl_vector_pipeline_value_extractor<dynfu::pixel> e(q_);
		auto && map=e(prev);

		buffer.resize(map.size(),q_);

		boost::compute::copy(map.begin(),map.end(),buffer.begin(),q_);

		return vn;

	}


}
