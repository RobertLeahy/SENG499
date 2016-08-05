#include <boost/compute/buffer.hpp>
#include <dynfu/boost_compute_detail_type_name_trait.hpp>
#include <dynfu/kinect_fusion_opencl_measurement_pipeline_block.hpp>
#include <dynfu/opencl_vector_pipeline_value.hpp>
#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <tuple>


namespace dynfu {
	
	
	static boost::compute::kernel get_bilateral_kernel (opencl_program_factory & opf) {
		
		auto p=opf("bilateral");
		return boost::compute::kernel(p,"bilateral_filter");
		
	}
	
	
	static boost::compute::kernel get_v_kernel (opencl_program_factory & opf) {
		
		auto p=opf("vertex_map");
		return boost::compute::kernel(p,"vertex_map");
		
	}
	
	
	static boost::compute::kernel get_n_kernel (opencl_program_factory & opf) {
		
		auto p=opf("normal_map");
		return boost::compute::kernel(p,"normal_map");
		
	}
	
	
	kinect_fusion_opencl_measurement_pipeline_block::kinect_fusion_opencl_measurement_pipeline_block (boost::compute::command_queue q, opencl_program_factory & opf, std::size_t window_size, float sigma_s, float sigma_r)
		:	ve_(std::move(q)),
			bilateral_kernel_(get_bilateral_kernel(opf)),
			v_kernel_(get_v_kernel(opf)),
			n_kernel_(get_n_kernel(opf)),
			v_(ve_.command_queue().get_context()),
			kbuf_(ve_.command_queue().get_context(),sizeof(Eigen::Matrix3f),CL_MEM_READ_ONLY)
	{
		
		bilateral_kernel_.set_arg(2,1.0f/sigma_s*sigma_s);
		bilateral_kernel_.set_arg(3,1.0f/sigma_r*sigma_r);
		std::uint32_t ws(window_size);	//	TODO: Make sure this doesn't overflow?
		bilateral_kernel_.set_arg(6,ws);
		bilateral_kernel_.set_arg(7,ws);
		
		v_kernel_.set_arg(2,kbuf_);
		
	}
	
	
	kinect_fusion_opencl_measurement_pipeline_block::value_type kinect_fusion_opencl_measurement_pipeline_block::operator () (
		depth_device::value_type::element_type & frame,
		std::size_t width,
		std::size_t height,
		Eigen::Matrix3f k,
		kinect_fusion_opencl_measurement_pipeline_block::value_type v
	) {
		
		//	Bilateral filtering
		//
		//	Kernel arguments are:
		//
		//	0:	Source
		//	1:	Destination
		//	2:	sigma_s^-2
		//	3:	sigma_r^-2
		//	4:	Width
		//	5:	Height
		auto && vec=ve_(frame);
		auto s=vec.size();
		auto q=ve_.command_queue();
		v_.resize(s,q);
		bilateral_kernel_.set_arg(0,vec);
		bilateral_kernel_.set_arg(1,v_);
		bilateral_kernel_.set_arg(4,std::uint32_t(width));	//	TODO: Make sure this doesn't overflow?
		bilateral_kernel_.set_arg(5,std::uint32_t(height));	//	TODO: Make sure this doesn't overflow?
		std::size_t extent []={width,height};
		q.enqueue_nd_range_kernel(bilateral_kernel_,2,nullptr,extent,nullptr);

		using pv_type=opencl_vector_pipeline_value<map_type::value_type>;
		if (!v) v=std::make_unique<pv_type>(q);
		auto && pv=dynamic_cast<pv_type &>(*v);
		auto && map=pv.vector();
		map.resize(s,q);
		
		//	Vertex map
		//
		//	Kernel arguments are:
		//
		//	0:	Source (bilaterally filtered image)
		//	1:	Destination (map)
		//	2:	K^-1
		if (k_!=k) {
			
			Eigen::Matrix3f k_inv=k.inverse();
			k_inv.transposeInPlace();
			q.enqueue_write_buffer(kbuf_,0,sizeof(k_inv),k_inv.data());
			k_=std::move(k);
			
		}
		
		v_kernel_.set_arg(0,v_);
		v_kernel_.set_arg(1,map);
		q.enqueue_nd_range_kernel(v_kernel_,2,nullptr,extent,nullptr);
		
		//	Normal map
		//
		//	Kernel arguments are:
		//
		//	0:	Map
		n_kernel_.set_arg(0,map);
		q.enqueue_nd_range_kernel(n_kernel_,2,nullptr,extent,nullptr);
		
		return v;
		
	}
	

}
