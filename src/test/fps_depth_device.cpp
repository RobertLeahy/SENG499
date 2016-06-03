#include <dynfu/fps_depth_device.hpp>


#include <dynfu/mock_depth_device.hpp>
#include <chrono>
#include <catch.hpp>


SCENARIO("fps_depth_device objects limit the rate at which frames may be acquired", "[dynfu][depth_device][fps_depth_device]") {
	
	GIVEN("An fps_depth_device object configured for 30FPS") {
		
		dynfu::mock_depth_device mock;
		mock.add();
		mock.add();
		dynfu::fps_depth_device fps(mock,30);
		std::chrono::milliseconds delay(33);
		
		WHEN("operator () is invoked") {
			
			using clock=std::chrono::high_resolution_clock;
			auto start=clock::now();
			fps();
			
			THEN("The call returns without delay") {
				
				auto delta=std::chrono::duration_cast<std::chrono::milliseconds>(clock::now()-start);
				CHECK(delta<delay);
				
			}
			
			AND_WHEN("operator() is invoked again") {
				
				fps();
				
				THEN("At least 1/30 seconds passes before the second invocation returns") {
					
					auto delta=std::chrono::duration_cast<std::chrono::milliseconds>(clock::now()-start);
					CHECK(delta>=delay);
					
				}
				
			}
			
		}
	
	}
	
	
}