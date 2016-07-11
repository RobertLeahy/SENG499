#include <dynfu/opencv_depth_device.hpp>


#include <catch.hpp>


SCENARIO("On creation dynfu::opencv_depth_device objects may be used to retrieve depth frames","[dynfu][opencv_depth_device][depth_device][!mayfail]") {

    GIVEN("A dynfu::opencv_depth_device object") {

        dynfu::opencv_depth_device dd;

        WHEN("A depth frame is retrieved") {

            auto pv=dd();
            REQUIRE(pv);
            auto && vec=pv->get();

            THEN("It is the correct size") {

                CHECK(vec.size()==(dd.width()*dd.height()));

            }

            THEN("The k matrix is correct") {

                auto k=dd.k();
                CHECK(k(0,0)==585.0f);
                CHECK(k(0,1)==0.0f);
                CHECK(k(0,2)==320.0f);
                CHECK(k(1,0)==0.0f);
                CHECK(k(1,1)==585.0f);
                CHECK(k(1,2)==240.0f);
                CHECK(k(2,0)==0.0f);
                CHECK(k(2,1)==0.0f);
                CHECK(k(2,2)==1.0f);

            }

        }

    }

}
