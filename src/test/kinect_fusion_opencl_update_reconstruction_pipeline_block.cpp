#include <dynfu/kinect_fusion_opencl_update_reconstruction_pipeline_block.hpp>

#include <boost/compute.hpp>
#include <dynfu/cpu_pipeline_value.hpp>
#include <dynfu/filesystem.hpp>
#include <dynfu/path.hpp>
#include <dynfu/file_system_opencl_program_factory.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <tuple>
#include <utility>
#include <vector>

#include <catch.hpp>


namespace {
	

	class fixture {
		
		private:
		
			static dynfu::filesystem::path cl_path () {
				
				dynfu::filesystem::path retr(dynfu::current_executable_parent_path());
				retr/="..";
				retr/="cl";
				
				return retr;
				
			}
		
		protected:
		
			boost::compute::device dev;
			boost::compute::context ctx;
			boost::compute::command_queue q;
			std::size_t width;
			std::size_t height;
			Eigen::Matrix3f k;
			Eigen::Matrix4f t_g_k;
			dynfu::file_system_opencl_program_factory fsopf;
			std::size_t tsdf_width;
			std::size_t tsdf_height;
			std::size_t tsdf_depth;
			
		public:
		
			fixture () : dev(boost::compute::system::default_device()), ctx(dev), q(ctx,dev), width(4), height(4), fsopf(cl_path(),ctx), tsdf_width(16), tsdf_height(16), tsdf_depth(16) {
					
				k << 0.585f, 0.0f, 16.0f,
					 0.0f, -0.585f, 16.0f,
					 0.0f, 0.0f, 1.0f;
					 
				t_g_k = Eigen::Matrix4f::Identity();
			
			}
		
	};
	

}


SCENARIO_METHOD(fixture, "A dynfu::kinect_fusion_opencl_update_reconstruction_pipeline_block implements the update_reconstruction phase of the kinect fusion pipeline on the GPU using OpenCL","[dynfu][update_reconstruction_pipeline_block][kinect_fusion_opencl_update_reconstruction_pipeline_block]") {
	
	GIVEN("A dynfu::kinect_fusion_opencl_update_reconstruction_pipeline_block") {		
		
		
		dynfu::kinect_fusion_opencl_update_reconstruction_pipeline_block kfourpb(q, fsopf, 100.0f, tsdf_width, tsdf_height, tsdf_depth);
		dynfu::cpu_pipeline_value<std::vector<float>> pv;
		std::vector<float> v{0.0f, 5.0f, 10.0f, 15.0f, 13.0f, 40.0f, 12.0f, 10.0f, 8.0f,19.0f,202.0f,102.0f,84.0f,293.0f,292.0f,293.0f};

		pv.emplace(std::move(v));
		
		
		THEN("Invoking it and downloading the produced TSDF does not fail") {
			
			auto f = [&](){
				
				
				
				auto && t=kfourpb(pv, width, height, k, t_g_k);
				auto && ts = t.buffer->get();
				
#ifdef GENERATE_GROUND_TRUTH
				std::size_t i(0);
				for (auto && t_ : ts) {
					
					std::cout << t_ <<",";
					
					i++;
					if ( i > 10) {
						std::cout << std::endl;
						i = 0;
					}
				}
#endif

#ifdef SLICE_VIEW	
				for (unsigned int z = 0; z < tsdf_width; z++) {
					
					std::cout << "Slice " << z << std::endl;
					
					for (unsigned int y = 0; y < tsdf_height; y++) {
						
						for (unsigned int x = 0; x < tsdf_depth; x++) {
							
							std::cout << ts.at(x + (tsdf_width+1)*y + ((tsdf_width + 1) * (tsdf_height + 1))*z) << ", ";
							
						}
						std::cout << std::endl;
					}
					
					std::cout << std::endl;

				}		
#endif
			};
			
			CHECK_NOTHROW(f());
			
		}
		
	}
	
}
