/**
 *	\file
 */


#pragma once

#include <dynfu/pose_estimation_pipeline_block.hpp>
#include <Eigen/Dense>
#include <cstddef>


namespace dynfu {
	
	
	/**
	 *	An implementation of a \ref pose_estimation_pipeline_block
	 *	that uses the Eigen math library to calculate a new sensor
	 *	pose estimation.
	 *
	 *	This class is highly experimental, highly unstable, and
	 *	extremely non-performant.  Do not use unless you know
	 *	exactly what you are doing and why.
	 *
	 *	\sa kinect_fusion
	 */
	class kinect_fusion_eigen_pose_estimation_pipeline_block: public pose_estimation_pipeline_block {

		private:
			float epsilon_d_;
			float epsilon_theta_;
			std::size_t frame_width_;
			std::size_t frame_height_;
			std::size_t numit_;
			Eigen::Matrix4f t_gk_initial_;

			Eigen::Matrix4f iterate(
				measurement_pipeline_block::vertex_value_type::element_type &,
				measurement_pipeline_block::normal_value_type::element_type &,
				measurement_pipeline_block::vertex_value_type::element_type *,
				measurement_pipeline_block::normal_value_type::element_type *,
				Eigen::Matrix3f,
				Eigen::Matrix4f,
				Eigen::Matrix4f
			);

		public:

			/**
			 *  Creates a new kinect_fusion_eigen_pose_estimation_pipeline_block.
			 * 
			 *  \param [in] epsilon_d
			 *      The rejection distance in Equation 17, \f$\epsilon_d\f$
			 *
			 *  \param [in] epsilon_theta
			 *      The normal rejection angle in Equation 17, \f$\epsilon_\theta\f$
			 *
			 *  \param [in] frame_width
			 *      The width of the depth frame
			 *
			 *  \param [in] frame_height
			 *      The height of the depth frame
			 *
			 *	\param [in] t_gk_initial
			 *		The \f$T_{g,k}\f$ to return from the first invocation.
			 *
			 *  \param [in] numit
			 *      The number of iterations to use
			 *
			 */
			kinect_fusion_eigen_pose_estimation_pipeline_block (
				float epsilon_d,
				float epsilon_theta,
				std::size_t frame_width,
				std::size_t frame_height,
				Eigen::Matrix4f t_gk_initial,
				std::size_t numit=15
			);

			virtual value_type operator () (
				measurement_pipeline_block::value_type::element_type &,
				measurement_pipeline_block::value_type::element_type *,
				Eigen::Matrix3f k,
				value_type t_gk_minus_one
			) override;


	};

	
}
