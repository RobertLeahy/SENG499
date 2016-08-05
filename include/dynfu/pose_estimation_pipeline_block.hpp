/**
 *	\file
 */


#pragma once


#include <Eigen/Dense>
#include <dynfu/pipeline_value.hpp>
#include <dynfu/measurement_pipeline_block.hpp>
#include <memory>
#include <stdexcept>

namespace dynfu {
	
	
	/**
	 *	An abstract base class which when implemented by
	 *	a derived class provides a sensor pose estimation
	 *	based on the vertex and normal maps of the current
	 *	frame, the global vertex and normal maps, and the
	 *	previous pose estimation, if any.
	 *
	 *	\sa kinect_fusion
	 */
	class pose_estimation_pipeline_block {
		
		
		public:
		
		
			/**
			 *	The 4x4 matrix \f$T_{g,k}\f$
			 */
			using sensor_pose_estimation_type=Eigen::Matrix4f;

			/**
			 *	A std::unique_ptr to a \ref pipeline_value
			 *	representing a sensor_pose_estimation_type
			 */
			using value_type=std::unique_ptr<pipeline_value<sensor_pose_estimation_type>>;


			pose_estimation_pipeline_block () = default;
			pose_estimation_pipeline_block (const pose_estimation_pipeline_block &) = delete;
			pose_estimation_pipeline_block (pose_estimation_pipeline_block &&) = delete;
			pose_estimation_pipeline_block & operator = (const pose_estimation_pipeline_block &) = delete;
			pose_estimation_pipeline_block & operator = (pose_estimation_pipeline_block &&) = delete;
			
			
			/**
			 *	Allows derived classes to be cleaned up through
			 *	pointer or reference to base.
			 */
			virtual ~pose_estimation_pipeline_block () noexcept;


			/**
			 *	Obtains a new sensor pose estimation \f$T_{g,k}\f$
			 *	from the current vertex and normal maps, registered
			 *	against the previously predicted vertex and normal
			 *	maps.
			 *
			 *	\param [in] map
			 *		A \ref pipeline_value which represents the vertex and
			 *		normal maps calculated for the current frame.
			 *	\param [in] prev_map
			 *		A pointer to a \ref pipeline_value which represents
			 *		the previous frame's simulated vertex and normal maps.
			 *		This pointer shall be \em nullptr for the first frame.
			 *	\param [in] k
			 *		Camera calibration matrix.
			 *	\param [in] t_gk_minus_one
			 *		The pose estimation calculated for the previous frame.
			 *		This std::unique_ptr will not manage a pointee for the
			 *		first frame.  The \ref pipeline_value managed by this
			 *		std::unique_ptr (if any) is guaranteed to be a value
			 *		previously returned by calling this function.  The
			 *		storage and object may be reused and returned by value
			 *		if advantageous for efficiency reasons.
			 *
			 *	\return
			 *		A \ref value_type object represernting an updated sensor
			 *		pose estimation, \f$T_{g,k}\f$.
			 */
			virtual value_type operator () (
				measurement_pipeline_block::value_type::element_type & map,
				measurement_pipeline_block::value_type::element_type * prev_map,
				Eigen::Matrix3f k,
				value_type t_gk_minus_one
			) = 0;


			/**
			*	An exception that indicates that a \ref pose_estimation_pipeline_block
			*	has lost the tracking of the scene.
			*
			*	Specifically, this is most often thrown if the projective data association
			*	does not find a minimum number of point-point correspondences.
			*/
			class tracking_lost_error: public std::runtime_error {

				public:

					using std::runtime_error::runtime_error;

			};
		
	
	};


}
