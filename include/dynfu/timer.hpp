/**
 *	\file
 */


#pragma once


#include <dynfu/high_resolution_clock.hpp>
#include <chrono>


namespace dynfu {
	
	
	/**
	 *	Represents a timer.
	 *
	 *	\tparam Clock
	 *		The clock the timer shall use to keep time.
	 */
	template <typename Clock>
	class basic_timer {
		
		
		public:
		
		
			using clock=Clock;
			using rep=typename clock::rep;
			using period=typename clock::period;
			using duration=typename clock::duration;
			using time_point=typename clock::time_point;
		
		
		private:
		
		
			time_point s_;
			
			
		public:
		
		
			basic_timer () : s_(clock::now()) {	}
			
			
			/**
			 *	Restarts the timer.
			 */
			void restart () {
				
				s_=clock::now();
				
			}
			
			
			/**
			 *	Determines the amount of time that has elapsed
			 *	since the timer was started or last restarted.
			 *
			 *	\return
			 *		The elapsed time.
			 */
			duration elapsed () const {
				
				return clock::now()-s_;
				
			}
			
			
			/**
			 *	Determines the amount of time that has elapsed
			 *	since the timer was started or last restarted
			 *	in milliseconds.
			 *
			 *	\return
			 *		The elapsed time in milliseconds.
			 */
			std::chrono::milliseconds elapsed_ms () const {
				
				return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed());
				
			}
			
			
			/**
			 *	Determines when the timer was last started or
			 *	restarted.
			 *
			 *	\return
			 *		The time when the timer was last started or
			 *		restarted.
			 */
			time_point started () const {
				
				return s_;
				
			}
		
		
	};
	
	
	/**
	 *	A \ref basic_timer which uses \ref high_resolution_clock
	 *	as its clock.
	 */
	using timer=basic_timer<high_resolution_clock>;
	
	
}
