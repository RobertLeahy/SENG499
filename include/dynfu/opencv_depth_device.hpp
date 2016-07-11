/**
 *	\file
 */


#include <dynfu/depth_device.hpp>
#include <Eigen/Dense>
#include <opencv2/videoio/videoio.hpp>
#include <cstddef>
#include <stdexcept>


namespace dynfu {


	/**
	 *	Uses OpenNI or OpenNI2 via cv::VideoCapture to interface
	 *	with a depth camera.
	 */
	class opencv_depth_device : public depth_device {


		private:


			cv::VideoCapture cap_;
			std::size_t width_;
			std::size_t height_;


			void open ();


		public:


			/**
			 *	Thrown from the constructor of \ref opencv_depth_camera
			 *	when it fails to open a depth camera.
			 */
			class open_camera_error : public std::runtime_error {


				public:


					using std::runtime_error::runtime_error;


			};


			/**
			 *	Thrown from \ref opencv_depth_camera::operator() when
			 *	it fails to capture a depth frame.
			 */
			class capture_error : public std::runtime_error {


				public:


					using std::runtime_error::runtime_error;


			};


			/**
			 *	Creates a new opencv_depth_device and attempts
			 *	to open a depth camera through OpenNI or OpenNI2.
			 */
			opencv_depth_device ();


			virtual value_type operator () (value_type v=value_type{}) override;
			virtual std::size_t width () const noexcept override;
			virtual std::size_t height () const noexcept override;
			virtual Eigen::Matrix3f k () const noexcept override;


	};


}
