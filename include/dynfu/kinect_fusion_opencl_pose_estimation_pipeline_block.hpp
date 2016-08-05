/**
 *	\file
 */


#pragma once


#include <boost/compute/buffer.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/kernel.hpp>
#include <dynfu/measurement_pipeline_block.hpp>
#include <dynfu/opencl_program_factory.hpp>
#include <dynfu/opencl_vector_pipeline_value.hpp>
#include <dynfu/optional.hpp>
#include <dynfu/pose_estimation_pipeline_block.hpp>
#include <Eigen/Dense>
#include <cstddef>


namespace dynfu {


	class kinect_fusion_opencl_pose_estimation_pipeline_block : public pose_estimation_pipeline_block {


		private:


			boost::compute::command_queue q_;
			boost::compute::kernel corr_;
			boost::compute::kernel parallel_sum_;
			boost::compute::kernel serial_sum_;
			boost::compute::buffer t_z_;
			boost::compute::buffer t_gk_prev_inverse_;
			boost::compute::buffer k_;
			boost::compute::buffer mats_;
			boost::compute::buffer mats_output_;
			opencl_vector_pipeline_value_extractor<measurement_pipeline_block::value_type::element_type::type::value_type> e_;
			opencl_vector_pipeline_value_extractor<measurement_pipeline_block::value_type::element_type::type::value_type> p_e_;
			float epsilon_d_;
			float epsilon_theta_;
			std::size_t frame_width_;
			std::size_t frame_height_;
			Eigen::Matrix4f t_gk_initial_;
			std::size_t numit_;
			std::size_t group_size_;


		public:


			kinect_fusion_opencl_pose_estimation_pipeline_block () = delete;


			kinect_fusion_opencl_pose_estimation_pipeline_block (
				opencl_program_factory & pf,
				boost::compute::command_queue q,
				float epsilon_d,
				float epsilon_theta,
				std::size_t frame_width,
				std::size_t frame_height,
				Eigen::Matrix4f t_gk_initial,
				std::size_t numit=15,
				std::size_t group_size=16
			);


			virtual value_type operator () (
				measurement_pipeline_block::value_type::element_type &,
				measurement_pipeline_block::value_type::element_type *,
				Eigen::Matrix3f,
				value_type
			) override;


	};


}
