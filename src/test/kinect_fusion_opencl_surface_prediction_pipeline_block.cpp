#include <dynfu/kinect_fusion_opencl_surface_prediction_pipeline_block.hpp>


#include <boost/compute.hpp>
#include <dynfu/cpu_pipeline_value.hpp>
#include <dynfu/kinect_fusion_opencl_update_reconstruction_pipeline_block.hpp>
#include <dynfu/msrc_file_system_depth_device.hpp>
#include <dynfu/file_system_depth_device.hpp>
#include <dynfu/file_system_opencl_program_factory.hpp>
#include <dynfu/opencl_depth_device.hpp>
#include <dynfu/filesystem.hpp>
#include <dynfu/path.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <cmath>
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

		fixture () : dev(boost::compute::system::default_device()), ctx(dev), q(ctx,dev), width(640), height(480), fsopf(cl_path(),ctx), tsdf_width(256), tsdf_height(256), tsdf_depth(256) {

				//needs to be a valid k matrix for a meaningful test
				k << 585.0f, 0.0f, 320.0f,
					 0.0f, 585.0f, 240.0f,
					 0.0f, 0.0f, 1.0f;

				//needs to be a valid t_g_k for meaningful test (?)
				t_g_k = Eigen::Matrix4f::Identity();
				t_g_k(0,3) = 1.5f;
				t_g_k(1,3) = 1.5f;
				t_g_k(2,3) = 1.5f;


			}

	};


}


static bool is_nan (const Eigen::Vector3f & v) noexcept {

	return std::isnan(v(0)) && std::isnan(v(1)) && std::isnan(v(2));

}


SCENARIO_METHOD(fixture, "A dynfu::kinect_fusion_opencl_surface_prediction_pipeline_block implements the surface prediction phase of the kinect fusion pipeline on the GPU using OpenCL","[dynfu][surface_prediction_pipeline_block][kinect_fusion_opencl_surface_prediction_pipeline_block]") {

	GIVEN("A dynfu::kinect_fusion_opencl_ruface_prediction_pipeline_block") {

		float mu = 0.1f;

		dynfu::kinect_fusion_opencl_update_reconstruction_pipeline_block kfourpb(q, fsopf, mu, tsdf_width, tsdf_height, tsdf_depth);
		dynfu::kinect_fusion_opencl_surface_prediction_pipeline_block kfosppb(q, fsopf, mu, tsdf_width);
		dynfu::cpu_pipeline_value<std::vector<float>> pv;

		dynfu::filesystem::path pp(dynfu::current_executable_parent_path());

		auto d=boost::compute::system::default_device();
		boost::compute::context ctx(d);
		boost::compute::command_queue q(ctx,d);

		dynfu::msrc_file_system_depth_device_frame_factory ff;
		dynfu::msrc_file_system_depth_device_filter f;
		dynfu::file_system_depth_device ddi(pp/".."/"data/test/tsdf_viewer/",ff,&f);
		dynfu::opencl_depth_device dd(ddi,q);

		dynfu::cpu_pipeline_value<Eigen::Matrix4f> t_g_k_pv;
		t_g_k_pv.emplace(t_g_k);
		auto && tsdf=kfourpb(*dd(), width, height, k, t_g_k_pv);

		THEN("Invoking it and downloading the produced V and N maps does not fail and produces V and N maps which are remotely sane") {

			dynfu::cpu_pipeline_value<std::vector<dynfu::pixel>> prev;
			prev.emplace();
			auto ptr = kfosppb(*tsdf.buffer, tsdf.width, tsdf.height, tsdf.depth, t_g_k_pv, k, prev, {});
			auto && map=ptr->get();

			Eigen::Vector3f nullv(0,0,0);
			auto nulls=std::count_if(map.begin(),map.end(),[&] (const auto & p) noexcept {	return !is_nan(p.n) && (p.n==nullv);	});
			CHECK(nulls==0);
			auto non_nan_v=std::count_if(map.begin(),map.end(),[&] (const auto & p) noexcept {	return !is_nan(p.v);	});
			CHECK(non_nan_v>0);
			auto non_nan_n=std::count_if(map.begin(),map.end(),[&] (const auto & p) noexcept {	return !is_nan(p.n);	});
			CHECK(non_nan_n>0);

		}

	}

}
