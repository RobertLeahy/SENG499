#include <dynfu/high_resolution_clock.hpp>


#ifdef _WIN32

#include <cstdint>
#include <system_error>
#include <windows.h>


namespace dynfu {


	static std::uint64_t get_freq () {

		std::uint64_t retr;
		if (!QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER *>(&retr))) throw std::system_error(
			GetLastError(),
			std::system_category()
		);

		return retr;

	}


	static const std::uint64_t freq=get_freq();


	high_resolution_clock::time_point high_resolution_clock::now () {

		std::uint64_t count;
		if (!QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER *>(&count))) throw std::system_error(
			GetLastError(),
			std::system_category()
		);

		count*=std::uintmax_t(period::num);
		std::uintmax_t den(period::den);
		if (den>freq) count*=den/freq;
		else count/=freq/den;
		
		return time_point(duration(count));

	}


}

#endif
