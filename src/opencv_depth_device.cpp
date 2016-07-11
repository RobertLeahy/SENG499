#include <dynfu/cpu_pipeline_value.hpp>
#include <dynfu/opencv_depth_device.hpp>
#include <opencv2/core/core.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>


namespace dynfu {


	void opencv_depth_device::open () {

		cap_.open(cv::CAP_OPENNI2);
		if (cap_.isOpened()) return;
		cap_.open(cv::CAP_OPENNI);
		if (cap_.isOpened()) return;
		throw open_camera_error("Could not open cv::VideoCapture with cv::CAP_OPENNI2 or cv::CAP_OPENNI");

	}


	opencv_depth_device::opencv_depth_device () {

		open();
		width_=cap_.get(cv::CAP_PROP_FRAME_WIDTH);
		height_=cap_.get(cv::CAP_PROP_FRAME_HEIGHT);

	}


	opencv_depth_device::value_type opencv_depth_device::operator () (value_type v) {
	
		cv::Mat depth;
		if (!cap_.grab()) throw capture_error("cv::VideoCapture::grab failed");
		if (!cap_.retrieve(depth,cv::CAP_OPENNI_DEPTH_MAP)) throw capture_error("cv::VideoCapture::retrieve with cv::CAP_OPENNI_DEPTH_MAP failed");
		
		//	Check the height and width to prevent undefined
		//	behaviour
		if (std::size_t(depth.rows)!=height_) {

			std::ostringstream ss;
			ss << "Expected depth frame to have " << height_ << " rows, got " << depth.rows;
			throw capture_error(ss.str());

		}

		if (std::size_t(depth.cols)!=width_) {

			std::ostringstream ss;
			ss << "Expected depth frame to have " << width_ << " columns, got " << depth.cols;
			throw capture_error(ss.str());

		}
		
		//	Initialize the pipeline_value if necessary
		using pipeline_value_type=cpu_pipeline_value<buffer_type>;
		if (!v) v=std::make_unique<pipeline_value_type>();
		auto && pv=dynamic_cast<pipeline_value_type &>(*v);
		auto && vec=pv.get_or_emplace();
		vec.reserve(height_*width_);
		vec.clear();

		//	Copy frame over
		for (std::size_t i=0;i<width_;++i) for (std::size_t j=0;j<height_;++j) {
			
			cv::Point p(i,j);
			auto curr=depth.at<std::uint16_t>(p);
			vec.push_back(
				(curr==0)
					?	std::numeric_limits<buffer_type::value_type>::quiet_NaN()
						//	OpenCV returns the depth in millimeters, we're returning
						//	meters
					:	(buffer_type::value_type(curr)/1000.0f)
			);

		}

		return v;

	}


	std::size_t opencv_depth_device::width () const noexcept {

		return width_;

	}


	std::size_t opencv_depth_device::height () const noexcept {

		return height_;

	}


	Eigen::Matrix3f opencv_depth_device::k () const noexcept {

		//	This K matrix was created by Jordan
		Eigen::Matrix3f retr(Eigen::Matrix3f::Zero());
		retr(0,0)=585.0f;
		retr(1,1)=585.0f;
		retr(2,2)=1.0f;
		retr(0,2)=320.0f;
		retr(1,2)=240.0f;

		return retr;

	}


}
