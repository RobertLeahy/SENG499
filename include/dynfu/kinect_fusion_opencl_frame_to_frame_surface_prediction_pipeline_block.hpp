/**
 *	\file
 */


#pragma once


#include <boost/compute/command_queue.hpp>
#include <dynfu/surface_prediction_pipeline_block.hpp>


namespace dynfu {


	class kinect_fusion_opencl_frame_to_frame_surface_prediction_pipeline_block : public surface_prediction_pipeline_block {


		private:


			boost::compute::command_queue q_;


		public:


			kinect_fusion_opencl_frame_to_frame_surface_prediction_pipeline_block () = delete;


			explicit kinect_fusion_opencl_frame_to_frame_surface_prediction_pipeline_block (boost::compute::command_queue q);


			virtual value_type operator () (
				update_reconstruction_pipeline_block::value_type::element_type &,
				std::size_t,
				std::size_t,
				std::size_t,
				pose_estimation_pipeline_block::value_type::element_type &,
				Eigen::Matrix3f,
				measurement_pipeline_block::value_type::element_type &,
				value_type
			) override;


	};


}
