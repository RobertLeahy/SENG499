#include <dynfu/kinect_fusion_opencl_pose_estimation_pipeline_block.hpp>


#include <boost/compute.hpp>
#include <dynfu/cpu_pipeline_value.hpp>
#include <dynfu/file_system_depth_device.hpp>
#include <dynfu/file_system_opencl_program_factory.hpp>
#include <dynfu/filesystem.hpp>
#include <dynfu/kinect_fusion_opencl_measurement_pipeline_block.hpp>
#include <dynfu/msrc_file_system_depth_device.hpp>
#include <dynfu/path.hpp>
#include <dynfu/pixel.hpp>
#include <dynfu/pose_estimation_pipeline_block.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
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


		public:


			boost::compute::device dev;
			boost::compute::context ctx;
			boost::compute::command_queue q;
			std::size_t width;
			std::size_t height;
			Eigen::Matrix3f k;
			Eigen::Matrix4f t_gk_initial;
			dynfu::file_system_opencl_program_factory pf;
			dynfu::msrc_file_system_depth_device_frame_factory ff;
			dynfu::msrc_file_system_depth_device_filter f;
			dynfu::file_system_depth_device dd;
			dynfu::kinect_fusion_opencl_measurement_pipeline_block mpb;
			std::size_t iterations;
			std::size_t group_size;


			std::vector<dynfu::pixel> to_global (std::vector<dynfu::pixel> ps) {

				std::transform(ps.begin(),ps.end(),ps.begin(),[&] (const auto & p) noexcept {

					Eigen::Vector4f vh(p.v(0),p.v(1),p.v(2),1.0f);
					vh=t_gk_initial*vh;
					Eigen::Vector3f v(vh(0),vh(1),vh(2));

					return dynfu::pixel{v,t_gk_initial.block<3,3>(0,0)*p.n};

				});

				return ps;

			}


			fixture ()
				:	dev(boost::compute::system::default_device()),
					ctx(dev),
					q(ctx,dev),
					width(640),
					height(480),
					t_gk_initial(Eigen::Matrix4f::Identity()),
					pf(cl_path(),ctx),
					dd(frames_path(),ff,&f),
					mpb(q,pf,16,2.0f,1.0f),
					iterations(15),	//	The default
					group_size(dev.get_info<std::size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE))
			{

				k <<	585.0f, 0.0f, 320.0f,
						0.0f, -585.0f, 240.0f,
						0.0f, 0.0f, 1.0f;
				
				t_gk_initial(0,3)=1.5f;
				t_gk_initial(1,3)=1.5f;
				t_gk_initial(2,3)=1.5f;

				//	Set to default (16) if CL_DEVICE_MAX_WORK_GROUP_SIZE is
				//	greater than that
				if (group_size>16) group_size=16;

			}


	};


}


SCENARIO_METHOD(fixture,"The constructor of dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block rejects degenerate group sizes","[pose_estimation_pipeline_block][kinect_fusion_opencl_pose_estimation_pipeline_block][dynfu]") {

	WHEN("dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block is constructed with a group size of zero an exception is thrown") {

		CHECK_THROWS_AS(dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block(pf,q,0.10f,std::sin(20.0f*3.14159f/180.0f),width,height,t_gk_initial,iterations,0),std::logic_error);

	}

	WHEN("dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block is constructed with a group size which does not evenly divide the frame size an exception is thrown") {

		CHECK_THROWS_AS(dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block(pf,q,0.10f,std::sin(20.0f*3.14159f/180.0f),width,height,t_gk_initial,iterations,19),std::logic_error);

	}

	WHEN("dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block is constructed with a group size which is greater than the frame size an exception is thrown") {

		CHECK_THROWS_AS(dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block(pf,q,0.10f,std::sin(20.0f*3.14159f/180.0f),width,height,t_gk_initial,iterations,600000),std::logic_error);

	}

}


SCENARIO_METHOD(fixture,"The constructor of dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block rejects requests to iterate zero times","[dynfu][pose_estimation_pipeline_block][kinect_fusion_opencl_pose_estimation_pipeline_block]") {

	WHEN("dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block is constructed with zero iterations") {

		CHECK_THROWS_AS(dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block(pf,q,0.10f,std::sin(20.0f*3.14159f/180.0f),width,height,t_gk_initial,0,group_size),std::logic_error);

	}

}


SCENARIO_METHOD(fixture,"dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block objects return the matrix passed to their constructor on their first invocation","[dynfu][pose_estimation_pipeline_block][kinect_fusion_opencl_pose_estimation_pipeline_block]") {

	GIVEN("A dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block object") {

		dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block pepb(pf,q,0.10f,std::sin(20.0f*3.14159f/180.0f),width,height,t_gk_initial,iterations,group_size);

		WHEN("It is invoked for the first time (i.e. passed NULL as the 3rd, 4th, and 5th arguments to dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block::operator ())") {

			auto frame_ptr=dd();
			auto m_ptr=mpb(*frame_ptr,width,height,k);
			auto ptr=pepb(*m_ptr,nullptr,k,{});

			THEN("The T_gk matrix provided to its constructor is returned") {

				CHECK(ptr->get()==t_gk_initial);

			}

		}

	}

}


SCENARIO_METHOD(fixture,"dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block objects derive a transformation matrix between two identical frames that is the identity matrix with some set, constant, initial translation","[dynfu][pose_estimation_pipeline_block][kinect_fusion_opencl_pose_estimation_pipeline_block]") {

	GIVEN("A dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block object") {

		dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block pepb(pf,q,0.10f,std::sin(20.0f*3.14159f/180.0f),width,height,t_gk_initial,iterations,group_size);

		WHEN("It is invoked on two identical sets of vertices and normals") {

			auto frame_ptr=dd();
			auto m_ptr=mpb(*frame_ptr,width,height,k);
			auto ptr=pepb(*m_ptr,nullptr,k,{});
			dynfu::cpu_pipeline_value<std::vector<dynfu::pixel>> pv;
			pv.emplace(to_global(m_ptr->get()));
			ptr=pepb(*m_ptr,&pv,k,std::move(ptr));

			THEN("The resulting T_gk matrix is an identity matrix with the translation components from the initial T_gk matrix") {

				auto t_gk=ptr->get();
				auto diff=t_gk-t_gk_initial;
				CHECK(diff.isZero());

			}

		}

	}


}




static bool is_bounded_by (float val, float a, float b) noexcept {

	if (val>std::max(a,b)) return false;
	if (val<std::min(a,b)) return false;
	return true;

}


SCENARIO_METHOD(fixture,"Between consecutive frames dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block reports only a small change in sensor pose","[dynfu][kinect_fusion_opencl_pose_estimation_pipeline_block][pose_estimation_pipeline_block]") {

	GIVEN("A dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block object") {

		dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block pepb(pf,q,0.10f,std::sin(20.0f*3.14159f/180.0f),width,height,t_gk_initial,iterations,group_size);

		WHEN("It is invoked on two consecutive frames") {

			auto frame_ptr=dd();
			auto m_ptr1=mpb(*frame_ptr,width,height,k);
			auto ptr=pepb(*m_ptr1,nullptr,k,{});
			frame_ptr=dd(std::move(frame_ptr));
			auto m_ptr2=mpb(*frame_ptr,width,height,k);
			dynfu::cpu_pipeline_value<std::vector<dynfu::pixel>> pv;
			pv.emplace(to_global(m_ptr2->get()));
			ptr=pepb(*m_ptr1,&pv,k,std::move(ptr));

			THEN("The resulting T_gk matrix is not simply the initial T_gk matrix") {

				CHECK(ptr->get()!=t_gk_initial);

			}

			THEN("The resulting T_gk matrix indicates a small change in camera pose") {

				auto t_gk=ptr->get();

				//	There should be minimal translation
				float min=1.49f;
				float max=1.51f;
				CHECK(is_bounded_by(t_gk(0,3),min,max));
				CHECK(is_bounded_by(t_gk(1,3),min,max));
				CHECK(is_bounded_by(t_gk(2,3),min,max));

				//	Bottom row should be == [0 0 0 1]
				CHECK(t_gk(3,0)==Approx(0.0f));
				CHECK(t_gk(3,1)==Approx(0.0f));
				CHECK(t_gk(3,2)==Approx(0.0f));
				CHECK(t_gk(3,3)==Approx(1.0f));

				//	The 3x3 upper left block should be a
				//	rotation matrix (there should be no scaling)
				//
				//	Note: There are matrix entries that are never
				//	accessed, we have a check for that later
				//
				//	Formulae from here: http://nghiaho.com/?page_id=846
				auto x_rot=std::atan2(t_gk(2,1),t_gk(2,2));
				auto y_rot=std::atan2(-t_gk(2,0),std::sqrt(std::pow(t_gk(2,1),2)+std::pow(t_gk(2,2),2)));
				auto z_rot=std::atan2(t_gk(1,0),t_gk(0,0));
				//	We expect small angles, say 5 degrees or less
				auto five_degrees=0.0872665f;
				CHECK(is_bounded_by(x_rot,five_degrees,-five_degrees));
				CHECK(is_bounded_by(y_rot,five_degrees,-five_degrees));
				CHECK(is_bounded_by(z_rot,five_degrees,-five_degrees));
				//	Now we recreate the rotation matrix and make
				//	sure it's approximately the same, this is how
				//	we check the entries we didn't touch above
				Eigen::Matrix3f x;
				x <<	1.0f, 0.0f, 0.0f,
						0.0f, std::cos(x_rot), -std::sin(x_rot),
						0.0f, std::sin(x_rot), std::cos(x_rot);
				Eigen::Matrix3f y;
				y <<	std::cos(y_rot), 0.0f, std::sin(y_rot),
						0.0f, 1.0f, 0.0f,
						-std::sin(y_rot), 0.0f, std::cos(y_rot);
				Eigen::Matrix3f z;
				z << 	std::cos(z_rot), -std::sin(z_rot), 0.0f,
						std::sin(z_rot), std::cos(z_rot), 0.0f,
						0.0f, 0.0f, 1.0f;
				Eigen::Matrix3f rot(z*y*x);
				Eigen::Matrix3f diff(t_gk.block<3,3>(0,0)-rot);
				CHECK(diff.isZero());

			}

		}

	}

}


SCENARIO_METHOD(fixture,"When a minimum number of point-to-point correspondences cannot be made dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block objects throw a dynfu::pose_estimation_pipeline_block::tracking_lost_error","[dynfu][kinect_fusion_opencl_pose_estimation_pipeline_block][pose_estimation_pipeline_block][!mayfail]") {

	GIVEN("A dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block object") {

		dynfu::kinect_fusion_opencl_pose_estimation_pipeline_block pepb(pf,q,0.10f,std::sin(20.0f*3.14159f/180.0f),width,height,t_gk_initial,iterations,group_size);

		THEN("Invoking it on two non-consecutive frames results in a dynfu::pose_estimation_pipeline_block::tracking_lost_error") {

			auto frame_ptr=dd();	//	Frame 00
			auto m_ptr1=mpb(*frame_ptr,width,height,k);
			frame_ptr=dd(std::move(frame_ptr));	//	Frame 01
			frame_ptr=dd(std::move(frame_ptr));	//	Frame 23
			auto m_ptr2=mpb(*frame_ptr,width,height,k);
			auto ptr=pepb(*m_ptr1,nullptr,k,{});
			dynfu::cpu_pipeline_value<std::vector<dynfu::pixel>> pv;
			pv.emplace(to_global(m_ptr2->get()));
			CHECK_THROWS_AS(pepb(*m_ptr1,&pv,k,std::move(ptr)),dynfu::pose_estimation_pipeline_block::tracking_lost_error);
			
		}

	}

}
