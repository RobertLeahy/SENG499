/**
 *	\file
 */


#pragma once


#ifndef _WIN32

#include <chrono>
namespace dynfu {	using high_resolution_clock=std::chrono::high_resolution_clock;	}

#else


#include <chrono>


namespace dynfu {


	class high_resolution_clock {


		public:


			using rep=std::chrono::nanoseconds::rep;
			using period=std::chrono::nanoseconds::period;
			using duration=std::chrono::nanoseconds;
			using time_point=std::chrono::time_point<high_resolution_clock>;
			static constexpr bool is_steady=true;


			static time_point now ();


	};


}


#endif
