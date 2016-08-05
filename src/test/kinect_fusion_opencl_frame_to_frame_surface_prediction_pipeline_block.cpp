#include <dynfu/kinect_fusion_opencl_frame_to_frame_surface_prediction_pipeline_block.hpp>


#include <boost/compute.hpp>
#include <dynfu/cpu_pipeline_value.hpp>
#include <dynfu/filesystem.hpp>
#include <dynfu/path.hpp>
#include <dynfu/pixel.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <tuple>
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

		public:

		fixture () : dev(boost::compute::system::default_device()), ctx(dev), q(ctx,dev) {	}

	};


}


SCENARIO_METHOD(fixture,"dynfu::kinect_fusion_opencl_frame_to_frame_surface_prediction_pipeline_block objects simply copy the provided vertex and normal map to their output","[dynfu][surface_prediction_pipeline_block][kinect_fusion_opencl_frame_to_frame_surface_prediction_pipeline_block]") {

	GIVEN("A dynfu::kinect_fusion_opencl_frame_to_frame_surface_prediction_pipeline_block object") {

		dynfu::kinect_fusion_opencl_frame_to_frame_surface_prediction_pipeline_block sppb(q);

		WHEN("It is invoked") {

			dynfu::cpu_pipeline_value<std::vector<float>> tsdf;
			tsdf.emplace();
			dynfu::cpu_pipeline_value<Eigen::Matrix4f> pose;
			pose.emplace(Eigen::Matrix4f::Zero());
			dynfu::cpu_pipeline_value<std::vector<dynfu::pixel>> map_pv;
			auto && map=map_pv.emplace();
			map.push_back({{1.0f,2.0f,3.0f},{4.0f,5.0f,6.0f}});

			auto ptr=sppb(tsdf,0,0,0,pose,Eigen::Matrix3f::Zero(),map_pv,dynfu::kinect_fusion_opencl_frame_to_frame_surface_prediction_pipeline_block::value_type{});

			THEN("The vertex and normal map provided is simply copied into the result") {

				auto && result=ptr->get();
				CHECK(std::equal(result.begin(),result.end(),map.begin(),map.end(),[&] (const auto & a, const auto & b) noexcept {

					return (a.v==b.v) && (a.n==b.n);

				}));

			}

		}

	}

}
