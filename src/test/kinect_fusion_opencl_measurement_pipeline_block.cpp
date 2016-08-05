#include <dynfu/kinect_fusion_opencl_measurement_pipeline_block.hpp>


#include <boost/compute.hpp>
#include <dynfu/camera.hpp>
#include <dynfu/cpu_pipeline_value.hpp>
#include <dynfu/file_system_depth_device.hpp>
#include <dynfu/file_system_opencl_program_factory.hpp>
#include <dynfu/filesystem.hpp>
#include <dynfu/msrc_file_system_depth_device.hpp>
#include <dynfu/path.hpp>
#include <dynfu/pixel.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>
#include <catch.hpp>


namespace {
	

	class fixture {
		
		
		private:
		
			static dynfu::filesystem::path curr_dir () {

				return dynfu::filesystem::path(dynfu::current_executable_parent_path());

			}


			static dynfu::filesystem::path cl_path () {

				auto retr=curr_dir();
				retr/="..";
				retr/="cl";

				return retr;

			}


			static dynfu::filesystem::path frames_path () {

				auto retr=curr_dir();
				retr/="..";
				retr/="data/test/kinect_fusion_eigen_pose_estimation_pipeline_block";

				return retr;

			}
		
		protected:
		
			boost::compute::device dev;
			boost::compute::context ctx;
			boost::compute::command_queue q;
			dynfu::msrc_file_system_depth_device_frame_factory ff;
			dynfu::msrc_file_system_depth_device_filter f;
			dynfu::file_system_depth_device dd;
			std::size_t width;
			std::size_t height;
			Eigen::Matrix3f k;
			dynfu::file_system_opencl_program_factory fsopf;
			
		public:
		
			fixture ()
				:	dev(boost::compute::system::default_device()),
					ctx(dev),
					q(ctx,dev),
					dd(frames_path(),ff,&f),
					width(4),
					height(4),
					fsopf(cl_path(),ctx)
			{
					
				k << 585.0f, 0.0f, 320.0f,
					 0.0f, -585.0f, 240.0f,
					 0.0f, 0.0f, 1.0f;
			
			}
		
	};
	

}


SCENARIO_METHOD(fixture, "A dynfu::kinect_fusion_opencl_measurement_pipeline_block implements the measurement phase of the kinect fusion pipeline on the GPU using OpenCL","[dynfu][measurement_pipeline_block][kinect_fusion_opencl_measurement_pipeline_block]") {
	
	GIVEN("A dynfu::kinect_fusion_opencl_measurement_pipeline_block") {		
		
		dynfu::kinect_fusion_opencl_measurement_pipeline_block kfompb(q, fsopf, 16, 2.0f, 1.0f);
		dynfu::cpu_pipeline_value<std::vector<float>> pv;
		std::vector<float> v{0.0f, 5.0f, 10.0f, 15.0f, 13.0f, 40.0f, 12.0f, 10.0f, 8.0f,19.0f,202.0f,102.0f,84.0f,293.0f,292.0f,293.0f};

		auto nan = std::numeric_limits<float>::quiet_NaN();

		std::vector<Eigen::Vector3f> vgt {
			
			{nan, nan, nan},
			{-2.7265, 2.05128, 5},
			{-5.44231, 4.1074, 10.0118},
			{-8.12818, 6.15382, 14.9999},
			{-7.10745, 5.30838, 12.9933},
			{-21.812, 16.3419, 40},
			{-6.51233, 4.89449, 11.9802},
			{-5.4252, 4.09029, 10.0118},
			{-4.37607, 3.25471, 8.00001},
			{-10.3607, 7.72991, 19},
			{-109.805, 82.1812, 202},
			{-55.2718, 41.4974, 102},
			{-45.9487, 34.0308, 84},
			{-159.709, 118.655, 292.883},
			{-158.844, 118.384, 292.213},
			{-158.707, 118.655, 292.883},
	
		};
		
		std::vector<Eigen::Vector3f> ngt {
			
			{nan, nan, nan},
			{0.42106, -0.735997, 0.53011},
			{0.746723, -0.368941, 0.553433},
			{nan, nan, nan},
			{0.72885, 0.673795, 0.121564},
			{-0.751723, 0.35527, -0.555604},
			{-0.196665, -0.940886, 0.275782},
			{nan, nan, nan},
			{0.454822, -0.710761, 0.536615},
			{0.577981, -0.596583, 0.5568},
			{-0.886228, -0.279082, -0.369748},
			{nan, nan, nan},
			{nan, nan, nan},
			{nan, nan, nan},
			{nan, nan, nan},
			{nan, nan, nan}
			
		};


		pv.emplace(std::move(v));
		
		WHEN("It is invoked") {
			
			auto ptr=kfompb(pv, width, height, k);
			auto && map=ptr->get();

			THEN("The returned normals are all either close to 1 in length or NaN") {
				
				for(auto && p : map) {

					bool result(((std::isnan(p.n(0)) && std::isnan(p.n(1)) && std::isnan(p.n(2))) || (p.n.norm() == Approx(1.0f))));
					CHECK(result);
				
				}
				
				
			}
			
			THEN("The returned vertices match the ground truth vertices within 0.001") {
				
				// Note: Eigen's default precision was too precise, here we only care about generally close
				CHECK ( std::equal(vgt.begin(),vgt.end(),map.begin(),map.end(),[] (const auto & a, const auto & b) noexcept {

					bool a_nan=std::isnan(a(0)) && std::isnan(a(1)) && std::isnan(a(2));
					if (a_nan) return std::isnan(b.v(0)) && std::isnan(b.v(1)) && std::isnan(b.v(2));
					
					return (a-b.v).isZero(0.001f);
				
				}) );

				
			}
			
			THEN("The returned normals match the ground truth normals within 0.001") {
				
				// Note: Eigen's default precision was too precise, here we only care about generally close
				CHECK ( std::equal(ngt.begin(),ngt.end(),map.begin(),map.end(),[] (const auto & a, const auto & b) noexcept {

					bool a_nan=std::isnan(a(0)) && std::isnan(a(1)) && std::isnan(a(2));
					if (a_nan) return std::isnan(b.n(0)) && std::isnan(b.n(1)) && std::isnan(b.n(2));
				
					return (a-b.n).isZero(0.001f);
				
				}) );

				
			}
			
		}
		
	}
	
}


SCENARIO_METHOD(fixture,"The vertices returned by a dynfu::kinect_fusion_opencl_measurement_pipeline_block may be deprojected back into pixel space","[dynfu][measurement_pipeline_block][kinect_fusion_opencl_measurement_pipeline_block]") {

	GIVEN("A dynfu::kinect_fusion_opencl_measurement_pipeline_block") {

		dynfu::kinect_fusion_opencl_measurement_pipeline_block kfompb(q, fsopf, 16, 2.0f, 1.0f);

		WHEN("It is run on a depth frame") {

			auto frame_ptr=dd();
			auto k=dd.k();
			auto h=dd.height();
			auto w=dd.width();
			auto ptr=kfompb(*frame_ptr,w,h,k);

			THEN("The vertices of the returned vertex map may be deprojected back into pixel space") {

				auto && map=ptr->get();
				std::size_t idx(0);
				for (int y=0;y<int(h);++y) {

					for (int x=0;x<int(w);++x,++idx) {

						auto v=map[idx].v;
						if (std::isnan(v(0))) continue;
						auto pair=dynfu::to_pixel(v,k);
						CHECK(pair.first(0)==x);
						CHECK(pair.first(1)==y);

					}

				}

			}

		}

	}

}
