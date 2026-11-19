// Unit test for classify_color with known BGR values.
#include "plate_reader/alpr.hpp"
#include <iostream>
#include <cassert>

using plate_reader::classify_color;  // it's in an anonymous-style inline... check

int main()
{
  // classify_color is defined inline in the header
  std::vector<std::pair<cv::Vec3d, std::string>> tests = {
    {{170, 164, 147}, "silver"},  // real image bumper mean
    {{205, 187, 191}, "silver"},
    {{40, 40, 40}, "black"},
    {{0, 0, 255}, "red"},
    {{255, 0, 0}, "blue"},
    {{0, 255, 255}, "yellow"},
    {{0, 180, 0}, "green"},
    {{250, 250, 250}, "white"},
    {{80, 60, 60}, "navy"},
  };
  int pass = 0, fail = 0;
  for (auto & [bgr, want] : tests) {
    std::string got = plate_reader::classify_color(bgr);
    if (got == want) { ++pass; std::cout << "PASS "; }
    else { ++fail; std::cout << "FAIL "; }
    std::cout << "BGR(" << bgr[0] << "," << bgr[1] << "," << bgr[2]
              << ") want=" << want << " got=" << got << "\n";
  }
  std::cout << pass << " passed, " << fail << " failed\n";
  return fail ? 1 : 0;
}
