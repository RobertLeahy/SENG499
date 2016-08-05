/**
 *	\file
 */


#pragma once


#include <boost/compute/buffer.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/kernel.hpp>
#include <dynfu/depth_device.hpp>
#include <dynfu/measurement_pipeline_block.hpp>
#include <dynfu/opencl_program_factory.hpp>
#include <dynfu/opencl_vector_pipeline_value.hpp>
#include <dynfu/optional.hpp>
#include <dynfu/pose_estimation_pipeline_block.hpp>
#include <dynfu/surface_prediction_pipeline_block.hpp>
#include <dynfu/update_reconstruction_pipeline_block.hpp>
#include <Eigen/Dense>
#include <cstddef>




namespace dynfu {
	
	
	/**
	 *	An \ref surface_prediction_pipeline_block implemented
	 *	as per the Kinect Fusion paper using OpenCL.
	 *
	 *	\sa kinect_fusion
	 */
	class kinect_fusion_opencl_surface_prediction_pipeline_block : public surface_prediction_pipeline_block {
		
		
		private:
			opencl_vector_pipeline_value_extractor<float> ve_;
			boost::compute::kernel raycast_kernel_;
			boost::compute::buffer t_g_k_buf_;
			boost::compute::buffer ik_buf_;
			optional<Eigen::Matrix4f> t_g_k_;
			optional<Eigen::Matrix3f> k_;
			float mu_;
			float tsdf_extent_;
			std::size_t tsdf_size_;
			std::size_t frame_width_;
			std::size_t frame_height_;
		
		public:
			
			kinect_fusion_opencl_surface_prediction_pipeline_block () = delete;
			
			/**
			 *	Creates a new kinect_fusion_opencl_surface_prediction_pipeline_block.
			 *
			 *	\param [in] q
			 *		A boost::compute::command_queue which the newly created
			 *		object shall use to dispatch OpenCL tasks.
			 *	\param [in] opf
			 *		An \ref opencl_program_factory which the newly created
			 *		object shall use to obtain OpenCL programs.
			 *	\param [in] mu
			 *		\f$\mu\f$: the truncation distance of the truncated signed distance function (TSDF).
			 *		Points greater than a distance of \f$\mu\f$ from the nearest surface interface are
			 *		truncated to the maximum distance of \f$\mu\f$. Non-visible points farther than \f$\mu\f$
			 *		from the surface are not measured.
			 *	\param [in] tsdf_size
			 *		The number of elements of the TSDF in each dimension (assume normal cuboid)
			 *	\param [in] tsdf_extent
			 *		The extent of the TSDF (in meters) in each dimensions (assume normal cuboid)
			 *	\param [in] frame_width_
			 *		The width of the depth frame.
			 *	\param [in] frame_height 
			 *		The height of the depth frame.
			 */
			kinect_fusion_opencl_surface_prediction_pipeline_block (				
				boost::compute::command_queue q,
				opencl_program_factory & opf,
				float mu,
				std::size_t tsdf_size=512,
				float tsdf_extent=3.0f,
				std::size_t frame_width=640,
				std::size_t frame_height=480
			);
			
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
