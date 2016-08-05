/**
 *	\file
 */


#pragma once


#include <Eigen/Dense>


namespace dynfu {


	/**
	 *	Represents a measured depth pixel.
	 *
	 *	\sa measurement_pipeline_block
	 */
	class pixel {


		public:


			/**
			 *	The measured vertex position in the relevant
			 *	coordinate space.
			 */
			Eigen::Vector3f v;
			/**
			 *	The measured normal in the relevant coordinate
			 *	space.
			 */
			Eigen::Vector3f n;


	};


	//	This makes sure that interop with OpenCL is smooth
	static_assert(sizeof(pixel)==(sizeof(float)*6U),"dynfu::pixel is not packed");


}
