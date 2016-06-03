#include <dynfu/opencl_depth_device.hpp>


#include <boost/compute.hpp>
#include <Eigen/Dense>
#include <dynfu/mock_depth_device.hpp>
#include <dynfu/opencl_vector_pipeline_value.hpp>
#include <algorithm>
#include <vector>
#include <catch.hpp>


SCENARIO("dynfu::opencl_depth_device objects wrap a dynfu::depth_device and upload its depth frames to the GPU","[dynfu][depth_device][opencl_depth_device]") {
	
	GIVEN("A dynfu::opencl_depth_device which wraps a dynfu::depth_device which does not upload its depth frames to the GPU") {
		
		dynfu::mock_depth_device mdd(Eigen::Matrix3f::Zero(),2,2);
		std::vector<float> v{1.0f,2.0f,3.0f,4.0f};
		mdd.add(v);	//	Copy so we have something to compare against
		auto dev=boost::compute::system::default_device();
		boost::compute::context ctx(dev);
		boost::compute::command_queue q(ctx,dev);
		dynfu::opencl_depth_device gpudd(mdd,q);
		
		WHEN("It is invoked") {
			
			auto pv=gpudd();
			
			THEN("The returned dynfu::pipeline_value is-a dynfu::opencl_vector_pipeline_value") {
				
				auto ptr=dynamic_cast<dynfu::opencl_vector_pipeline_value<float> *>(pv.get());
				REQUIRE(ptr);
				
				AND_THEN("The frame is available on the GPU") {
					
					auto && ve=ptr->vector();
					q.finish();	//	In case any operation is still pending
					CHECK(std::equal(v.begin(),v.end(),ve.begin(),ve.end()));
					
				}
				
			}
			
			THEN("The frame is available on the CPU") {
				
				auto && ve=pv->get();
				CHECK(std::equal(v.begin(),v.end(),ve.begin(),ve.end()));
				
			}
			
		}
		
	}
	
}
