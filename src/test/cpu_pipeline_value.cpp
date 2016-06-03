#include <dynfu/cpu_pipeline_value.hpp>


#include <catch.hpp>


SCENARIO("Values may be provided and retrieved through a dynfu::cpu_pipeline_value<T>","[dynfu][pipeline_value][cpu_pipeline_value]") {
	
	GIVEN("A dynfu::cpu_pipeline_value<T>") {
		
		dynfu::cpu_pipeline_value<int> pv;
		
		WHEN("A value is emplaced therein") {
			
			auto && v=pv.emplace(5);
			
			THEN("dynfu::cpu_pipeline_value::get returns a reference to that same object") {
				
				CHECK(&v==&pv.get());
				
			}
			
			THEN("The correct value is created") {
				
				CHECK(v==5);
				
			}
			
		}
		
	}
	
}
