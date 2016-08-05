/**
 *	\file
 */


#pragma once


#include <dynfu/measurement_pipeline_block.hpp>
#include <dynfu/pipeline_value.hpp>
#include <dynfu/pixel.hpp>
#include <dynfu/pose_estimation_pipeline_block.hpp>
#include <dynfu/update_reconstruction_pipeline_block.hpp>
#include <Eigen/Dense>
#include <cstddef>
#include <memory>
#include <tuple>
#include <vector>


namespace dynfu {
	
	
	/**
	 *	An abstract base class which when implemented in
	 *	a derived class takes as input the truncated
	 *	signed distance function (TSDF) and previous camera
	 *	pose estimation and outputs the new global vertex
	 *	and normal maps.
	 *
	 *	\sa kinect_fusion
	 */
	class surface_prediction_pipeline_block {
		
		
		public:


			/**
			 *	A collection of vertices.
			 *
			 *	Each vertex in a vertex map has a corresponding
			 *	normal in the associated normal map.
			 */
			using map_type=std::vector<pixel>;
			/**
			 *	The type this pipeline block generates.
			 *
			 *	A tuple whose first element is a vertex map and whose
			 *	second element is the associated normal map.
			 */
			using value_type=std::unique_ptr<pipeline_value<map_type>>;


			surface_prediction_pipeline_block () = default;
			surface_prediction_pipeline_block (const surface_prediction_pipeline_block &) = delete;
			surface_prediction_pipeline_block (surface_prediction_pipeline_block &&) = delete;
			surface_prediction_pipeline_block & operator = (const surface_prediction_pipeline_block &) = delete;
			surface_prediction_pipeline_block & operator = (surface_prediction_pipeline_block &&) = delete;
			
			
			/**
			 *	Allows derived classes to be cleaned up
			 *	through pointer or reference to base.
			 */
			virtual ~surface_prediction_pipeline_block () noexcept;


			/**
			 *	Calculates the predicted vertex and normal maps.
			 *
			 *	\param [in] tsdf
			 *		A \ref pipeline_value which represents the raw
			 *		values of the TSDF.
			 *	\param [in] width
			 *		The width of the TSDF.
			 *	\param [in] height
			 *		The height of the TSDF.
			 *	\param [in] depth
			 *		The depth of the TSDF.
			 *	\param [in] t_g_k
			 *		The current pose estimation.
			 *	\param [in] k
			 *		The camera calibration matrix.
			 *	\param [in] vn
			 *		The vertex and normal maps from the current
			 *		frame.
			 *	\param [in] map
			 *		The value previously returned by this
			 *		pipeline block if it has already been invoked,
			 *		or a tuple of std::unique_ptr objects which
			 *		do not manage a pointee.  If the std::unique_ptr
			 *		objects in this tuple manage pointees they must
			 *		be pointees previously generated by a call to this
			 *		function.
			 *
			 *	\return
			 *		The predicted vertex and normal maps.
			 */
			virtual value_type operator () (
				update_reconstruction_pipeline_block::value_type::element_type & tsdf,
				std::size_t width,
				std::size_t height,
				std::size_t depth,
				pose_estimation_pipeline_block::value_type::element_type & t_g_k,
				Eigen::Matrix3f k,
				measurement_pipeline_block::value_type::element_type & vn,
				value_type map
			) = 0;
		
		
	};
	
	
}
