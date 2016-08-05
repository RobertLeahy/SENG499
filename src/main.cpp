#include <boost/compute.hpp>
#include <boost/program_options.hpp>
#include <dynfu/depth_device.hpp>
#include <dynfu/file_system_depth_device.hpp>
#include <dynfu/file_system_opencl_program_factory.hpp>
#include <dynfu/filesystem.hpp>
#include <dynfu/kinect_fusion.hpp>
#include <dynfu/kinect_fusion_opencl_surface_prediction_pipeline_block.hpp>
#include <dynfu/kinect_fusion_opencl_measurement_pipeline_block.hpp>
#include <dynfu/kinect_fusion_opencl_pose_estimation_pipeline_block.hpp>
#include <dynfu/kinect_fusion_opencl_update_reconstruction_pipeline_block.hpp>
#include <dynfu/libigl.hpp>
#include <dynfu/msrc_file_system_depth_device.hpp>
#include <dynfu/opencl_depth_device.hpp>
#include <dynfu/opencv_depth_device.hpp>
#include <dynfu/optional.hpp>
#include <dynfu/path.hpp>
#include <dynfu/timer.hpp>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>


namespace {


	class program_options {


		public:


			dynfu::optional<dynfu::filesystem::path> dataset;
			dynfu::optional<std::size_t> max_frames;
			dynfu::optional<dynfu::filesystem::path> save_off;
			dynfu::optional<dynfu::filesystem::path> read_off;


	};


}


static dynfu::optional<program_options> get_program_options (int argc, char ** argv) {

	boost::program_options::options_description desc("Command line flags");
	desc.add_options()("dataset",boost::program_options::value<std::string>(),"Path to depth frames")
		("max-frames",boost::program_options::value<std::size_t>(),"Max frames to process")
		("save-off",boost::program_options::value<std::string>(),"Save the generated mesh to this .off file")
		("read-off",boost::program_options::value<std::string>(),"Read the generated mesh to this .off file and exit")
		("help,?","Display usage information");

	boost::program_options::variables_map vm;
	boost::program_options::store(boost::program_options::parse_command_line(argc,argv,desc),vm);
	boost::program_options::notify(vm);

	if (vm.count("help")) {

		std::cout << desc << std::endl;
		return dynfu::nullopt;

	}

	program_options retr;

	if (vm.count("dataset")) retr.dataset.emplace(vm["dataset"].as<std::string>());
	if (vm.count("max-frames")) retr.max_frames.emplace(vm["max-frames"].as<std::size_t>());
	if (vm.count("save-off")) retr.save_off.emplace(vm["save-off"].as<std::string>());
	if (vm.count("read-off")) retr.read_off.emplace(vm["read-off"].as<std::string>());

	return retr;

}


static void display_off(dynfu::filesystem::path path) {

    Eigen::MatrixXi SF;
    Eigen::MatrixXd SV;

    dynfu::libigl::readOFF(path.string(), SV, SF);
    dynfu::libigl::viewer(SV,SF);

}

static void main_impl (int argc, char ** argv) {

	auto opt=get_program_options(argc,argv);
	if (!opt) return;
	auto && options=*opt;
	

	if (options.read_off) {

		display_off(*options.read_off);
		return;

	}

	auto d=boost::compute::system::default_device();
	boost::compute::context ctx(d);
	boost::compute::command_queue q(ctx,d);

	dynfu::optional<dynfu::msrc_file_system_depth_device_frame_factory> ff;
	dynfu::optional<dynfu::msrc_file_system_depth_device_filter> f;
	dynfu::optional<dynfu::file_system_depth_device> ddi;
	dynfu::optional<dynfu::opencv_depth_device> ddocv;
	dynfu::depth_device * ddp;

	if (options.dataset) {

		ff.emplace();
		f.emplace();
		ddi.emplace(*options.dataset,*ff,&*f);
		ddp=&*ddi;

	} else {

		ddocv.emplace();
		ddp=&*ddocv;

	}

	dynfu::opencl_depth_device dd(*ddp,q);

	dynfu::filesystem::path pp(dynfu::current_executable_parent_path());

	float tsdf_extent = 3.0;
	std::size_t tsdf_size(256);
	float mu = 0.03f;

	dynfu::file_system_opencl_program_factory opf(pp/".."/"cl",ctx);
	dynfu::kinect_fusion_opencl_measurement_pipeline_block mpb(q,opf,20,2.0f,1.0f);
	Eigen::Matrix4f t_g_k(Eigen::Matrix4f::Identity());
	t_g_k(0,3)=1.5f;
	t_g_k(1,3)=1.5f;
	t_g_k(2,3)=1.5f;
	dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block pepb(opf,q,0.1f,std::sin(20.0f*3.14159254f/180.0f),dd.width(),dd.height(),t_g_k,15);
	dynfu::kinect_fusion_opencl_update_reconstruction_pipeline_block urpb(q,opf,mu);
	dynfu::kinect_fusion_opencl_surface_prediction_pipeline_block sppb(q,opf,mu,tsdf_size, tsdf_extent, dd.width(), dd.height());


	dynfu::kinect_fusion kf;
	kf.depth_device(dd);
	kf.measurement_pipeline_block(mpb);
	kf.pose_estimation_pipeline_block(pepb);
	kf.update_reconstruction_pipeline_block(urpb);
	kf.surface_prediction_pipeline_block(sppb);

	std::size_t total(0);
	std::size_t frames(0);
	while (ddi) {
			
		dynfu::timer t;
		kf();
		//	We do this because we're not actually consuming
		//	the generated values, in the future this shouldn't
		//	be necessary as the pipeline_value objects we consume
		//	will do this or equivalent internally as they'll have
		//	to wait for their offered values to become available
		//	on the GPU before they download them
		q.finish();
		auto e=t.elapsed_ms();
		std::cout << "Depth frame took " << std::chrono::duration_cast<std::chrono::milliseconds>(kf.depth_device_elapsed()).count() << "ms" << std::endl;
		std::cout << "Measurement took " << std::chrono::duration_cast<std::chrono::milliseconds>(kf.measurement_pipeline_block_elapsed()).count() << "ms" << std::endl;
		std::cout << "Pose estimation took " << std::chrono::duration_cast<std::chrono::milliseconds>(kf.pose_estimation_pipeline_block_elapsed()).count() << "ms" << std::endl;
		std::cout << "Updating reconstruction took " << std::chrono::duration_cast<std::chrono::milliseconds>(kf.update_reconstruction_pipeline_block_elapsed()).count() << "ms" << std::endl;
		std::cout << "Surface prediction took " << std::chrono::duration_cast<std::chrono::milliseconds>(kf.surface_prediction_pipeline_block_elapsed()).count() << "ms" << std::endl;
		std::cout << "Frame took " << e.count() << "ms" << std::endl;
		total+=e.count();

		if (options.max_frames) {

			std::cout << "Finished frame " << ++frames << " / " << *options.max_frames << std::endl;
			if (frames == *options.max_frames) break;

		}
		
	}
	
	std::cout << "Frames: " << frames << std::endl;
	std::size_t avg=(frames==0) ? 0 : (total/frames);
	std::cout << "Average time per frame: " << avg << "ms" << std::endl;
	
	auto && tsdf = kf.truncated_signed_distance_function().get();

	float px,py,pz;
    Eigen::MatrixXi SF;
    Eigen::MatrixXd SV;
    Eigen::VectorXd S_(tsdf_size*tsdf_size*tsdf_size);
    Eigen::MatrixXd GV(tsdf_size*tsdf_size*tsdf_size,3);
    std::size_t abs_idx(0);
	std::size_t num_nans(0);

    for(std::size_t zi(0); zi < tsdf_size; ++zi) {

        pz = (float(zi)+0.5f) * tsdf_extent/tsdf_size;

        for(std::size_t yi(0); yi < tsdf_size; ++yi) {

            py = (float(yi)+0.5f) * tsdf_extent/tsdf_size;

            for(std::size_t xi(0); xi < tsdf_size; ++xi) {

                px = (float(xi)+0.5f) * tsdf_extent/tsdf_size;
               
                std::size_t vox_idx(xi + tsdf_size * (yi + tsdf_size * zi));

                GV.row(vox_idx) = Eigen::RowVector3d(px,py,pz);

				if (std::isnan(tsdf[vox_idx])) {
					S_(abs_idx) = 1.0f;
					num_nans++;
				} else {
 					S_(abs_idx) = tsdf[vox_idx];
				}
               
                abs_idx++;

            }

        }

        std::cout << "Export Progress: "  << (zi + 1) << " / " << tsdf_size << std::endl;

    }

	std::cout << "Num NaN: " << num_nans << std::endl;
    std::cout << "Running Marching Cubes..." << std::endl;

    dynfu::libigl::marching_cubes(S_,GV,tsdf_size,tsdf_size,tsdf_size,SV,SF);

	if (options.save_off) {
		std::cout << "Writing mesh to " << options.save_off->string() << std::endl;
		dynfu::libigl::writeOFF(options.save_off->string(), SV, SF);
	}

    dynfu::libigl::viewer(SV,SF);

}


int main (int argc, char ** argv) {
	
	try {
		
		try {
			
			main_impl(argc,argv);
			
		} catch (const std::exception & ex) {
			
			std::cerr << "ERROR: " << ex.what() << std::endl;
			throw;
			
		} catch (...) {
			
			std::cerr << "ERROR" << std::endl;
			throw;
			
		}
		
	} catch (...) {
		
		return EXIT_FAILURE;
		
	}
	
}