/**
 *	\file
 */


#pragma once


#include <boost/compute.hpp>
#include <dynfu/pixel.hpp>
#include <Eigen/Dense>


//	This is evil black magic
namespace boost {
	
	
	namespace compute {
		
		
		namespace detail {
			
			
			template <>
			struct type_name_trait<Eigen::Vector3f> {
				
				
				static const char * value () noexcept {
					
					return "Eigen::Vector3f";
					
				}
				
				
			};


			template <>
			struct type_name_trait<dynfu::pixel> {
				
				
				static const char * value () noexcept {
					
					return "dynfu::pixel";
					
				}
				
				
			};
			
			
		}
		
		
	}
	
	
}
