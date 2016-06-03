#include <dynfu/opencl_vector_pipeline_value.hpp>


#include <boost/compute.hpp>
#include <dynfu/cpu_pipeline_value.hpp>
#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>
#include <catch.hpp>


namespace {
	
	
	class fixture {
		
		
		protected:
		
		
			boost::compute::device dev;
			boost::compute::context ctx;
			boost::compute::command_queue q;
			
			
		public:
		
		
			fixture () : dev(boost::compute::system::default_device()), ctx(dev), q(ctx,dev) {	}
		
		
	};
	
	
}


SCENARIO_METHOD(fixture,"dynfu::opencl_vector_pipeline_value objects keep the CPU copy up to date with the GPU copy as needed","[dynfu][pipeline_value][opencl_pipeline_value][opencl_vector_pipeline_value]") {
	
	GIVEN("An empty dynfu::opencl_vector_pipeline_value") {
		
		dynfu::opencl_vector_pipeline_value<int> pv(q);
		
		THEN("The CPU copy is empty") {
			
			CHECK(pv.get().empty());
			
		}
		
		WHEN("Data is transferred to the underlying boost::compute::vector retrieved through the dynfu::opencl_vector_pipeline_value::vector accessor") {
			
			auto && gpu=pv.vector();
			std::vector<int> cpu{1,2,3,4};
			boost::compute::copy(cpu.begin(),cpu.end(),std::back_inserter(gpu),q);
			
			THEN("The CPU copy is updated when it is retrieved again") {
				
				auto && v=pv.get();
				CHECK(std::equal(v.begin(),v.end(),cpu.begin(),cpu.end()));
				
			}
			
		}
		
	}
	
}


SCENARIO_METHOD(fixture,"dynfu::opencl_vector_pipeline_value_extractor objects allow pipeline_value objects to be transparently used on the GPU","[dynfu][pipeline_value][opencl_vector_pipeline_value_extractor]") {
	
	GIVEN("A dynfu::opencl_vector_pipeline_value_extractor") {
		
		dynfu::opencl_vector_pipeline_value_extractor<int> e(q);
		
		GIVEN("A dynfu::opencl_vector_pipeline_value associated with the same boost::compute::command_queue") {
			
			dynfu::opencl_vector_pipeline_value<int> pv(q);
			
			WHEN("The dynfu::opencl_vector_pipeline_value_extractor is invoked and given the dynfu::opencl_vector_pipeline_value") {
				
				auto && v=e(pv);
				
				THEN("The boost::compute::vector wrapped by the dynfu::opencl_vector_pipeline_value is returned") {
					
					CHECK(&v==&pv.vector());
					
				}
				
			}
			
		}
		
		GIVEN("A dynfu::opencl_vector_pipeline_value associated with a different boost::compute::context") {
			
			boost::compute::context n_ctx(dev);
			REQUIRE(n_ctx!=ctx);
			boost::compute::command_queue n_q(n_ctx,dev);
			REQUIRE(n_q!=q);
			dynfu::opencl_vector_pipeline_value<int> pv(n_q);
			int arr []={1,2,3};
			pv.vector().assign(std::begin(arr),std::end(arr),n_q);
			
			WHEN("The dynfu::opencl_vector_pipeline_value_extractor is invoked and given the dynfu::opencl_vector_pipeline_value") {
				
				auto && v=e(pv);
				
				THEN("A different boost::compute::vector than that wrapped by the dynfu::opencl_vector_pipeline_value is returned") {
					
					CHECK(&v!=&pv.vector());
					
					AND_THEN("The returned boost::compute::vector contains the same data as is contained by the boost::compute::vector wrapped by the dynfu::opencl_vector_pipeline_value") {
						
						CHECK(std::equal(std::begin(arr),std::end(arr),v.begin(),v.end()));
						
					}
					
				}
				
			}
			
		}
		
		GIVEN("A dynfu::cpu_pipeline_value") {
			
			dynfu::cpu_pipeline_value<std::vector<int>> pv;
			std::vector<int> v{1,2,3};
			pv.emplace(std::move(v));
			
			WHEN("The dynfu::opencl_vector_pipeline_value_extractor is invoked and given the dynfu::cpu_pipeline_value") {
				
				auto && v=e(pv);
				
				THEN("The returned boost::compute::vector contains the same data as the std::vector wrapped by the dynfu::cpu_pipeline_value") {
					
					auto && cpu_v=pv.get();
					CHECK(std::equal(v.begin(),v.end(),cpu_v.begin(),cpu_v.end()));
					
				}
				
			}
			
		}
		
	}
	
}
