#include <dynfu/msrc_file_system_depth_device.hpp>


#include <dynfu/filesystem.hpp>
#include <Eigen/Dense>
#include <dynfu/path.hpp>
#include <cmath>
#include <cstddef>
#include <unordered_set>
#include <catch.hpp>


SCENARIO("msrc_file_system_depth_device loads depth information from MSRC 7Scene files in the file system", "[dynfu][depth_device][file_system_depth_device][msrc_file_system_depth_device]") {
	
	dynfu::filesystem::path test_data_path(dynfu::current_executable_parent_path());
	test_data_path/="..";
	test_data_path/="data/test/msrc_file_system_depth_device";
	
	GIVEN("A valid msrc_file_system_depth_device") {
		
		dynfu::msrc_file_system_depth_device_frame_factory fac;
		dynfu::msrc_file_system_depth_device_filter filt;
		dynfu::file_system_depth_device fsdd(std::move(test_data_path),fac, &filt);
		

		WHEN("Returning the width and height") {

			auto w = fsdd.width();
			auto h = fsdd.height();

			THEN("A width of 640 is returned") {

				CHECK(w==640U);

			}

			THEN("A height of 480 is returned") {

				CHECK(h==480U);

			}

		}

		WHEN("It is invoked on the test image data/test/msrc_file_system_depth_device/frame-000000.depth.png") {

			auto ptr=fsdd();
			auto && frame=ptr->get();
			
			THEN("A frame of the correct size is returned") {
				
				auto s=frame.size();
				REQUIRE(s==640U*480U);
				
				AND_THEN("Invalid depth values are transformed into qNaNs") {
					
					std::unordered_set<std::size_t> is{
						19819,
						19820,
						19821,
						20458,
						20459,
						21096,
						21732,
						21733,
						21734,
						21735,
						21736,
						22372,
						22373,
						22374,
						22375,
						23011,
						23012,
						28787,
						28788,
						28789,
						28790,
						29421,
						29422,
						29425,
						29426,
						29427,
						29428,
						29429,
						29430,
						30047,
						30048,
						30049,
						30050,
						30051,
						30060,
						30061,
						30062,
						30063,
						30064,
						30065,
						30066,
						30067,
						30068,
						30687,
						30688,
						30689
					};
					for (std::size_t i=0;i<s;++i) CHECK(std::isnan(frame[i])==(is.count(i)!=0));
					
				}
				
			}

		}
	
	}
	
	
	GIVEN("An invalid msrc_file_system_depth_device") {

		
		test_data_path/="garbage";
		dynfu::msrc_file_system_depth_device_frame_factory fac;
		dynfu::file_system_depth_device fsdd(std::move(test_data_path),fac);
		
		THEN("Invoking it throws an exception") {

			CHECK_THROWS_AS(fsdd(), std::runtime_error);
			
		}


	}

}
