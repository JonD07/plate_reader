// Test harness for the C++ ALPR pipeline (no ROS2 in this sandbox).
// Uses the real src/alpr.cpp; prints results identical in format to the
// Python prototype so outputs can be compared directly.
#include "plate_reader/alpr.hpp"

#include <chrono>
#include <iostream>
#include <string>

int main(int argc, char ** argv)
{
  if (argc < 2) {
    std::cerr << "usage: alpr_test <tessdata_dir> <image> [image...]\n";
    return 1;
  }
  std::string tessdata = argv[1];
  plate_reader::AlprEngine engine(tessdata);

  for (int i = 2; i < argc; ++i) {
    cv::Mat img = cv::imread(argv[i], cv::IMREAD_COLOR);
    if (img.empty()) {
      std::cout << argv[i] << " -> cannot read" << std::endl;
      continue;
    }
    auto t0 = std::chrono::steady_clock::now();
    plate_reader::PlateResult r = engine.read_plate(img);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << argv[i] << " -> "
              << (r.success ? "OK" : "FAIL")
              << " plate='" << r.plate_number << "'"
              << " conf=" << r.confidence
              << " state='" << r.issuing_state << "'"
              << " sconf=" << r.state_confidence
              << " raw='" << r.state_raw << "'"
              << " color=" << r.car_color
              << " msg='" << r.message << "'"
              << " (" << ms << " ms)"
              << std::endl;
  }
  return 0;
}
