// plate_reader/plate_reader_node.cpp — ROS2 service node that reads a
// license plate from an image file and returns the plate number plus
// confidence, issuing state and car color.
//
// Service: plate_reader/read_plate  (plate_reader/srv/ReadPlate)
//   request:  image_path  (string) — relative or absolute path to a .png
//   response: success, plate_number, confidence, issuing_state,
//             state_confidence, state_raw, car_color, message
//
// Relative paths are resolved against (first match wins):
//   1. the path as given (cwd of the node)
//   2. $PLATE_READER_IMAGE_DIR if set
//   3. $HOME/plate_images/
//   4. /tmp/plate_images/

#include "plate_reader/alpr.hpp"

#include <rclcpp/rclcpp.hpp>
#include <plate_reader/srv/read_plate.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>

namespace fs = std::filesystem;
using plate_reader::srv::ReadPlate;

namespace plate_reader
{

class PlateReaderNode : public rclcpp::Node
{
public:
  explicit PlateReaderNode(const rclcpp::NodeOptions & options)
  : Node("plate_reader", options)
  {
    // tessdata location: param > env > system default (nullptr)
    std::string tessdata = declare_parameter<std::string>(
      "tessdata_path",
      std::string(""));
    if (tessdata.empty()) {
      const char * env = std::getenv("TESSDATA_PREFIX");
      if (env) tessdata = env;
    }
    try {
      engine_ = std::make_unique<AlprEngine>(tessdata);
    } catch (const std::exception & e) {
      RCLCPP_FATAL(get_logger(),
                   "Tesseract init failed: %s — is eng.traineddata "
                   "installed? (apt install tesseract-ocr)", e.what());
      throw;
    }

    srv_ = create_service<ReadPlate>(
      "read_plate",
      [this](const std::shared_ptr<ReadPlate::Request> request,
             std::shared_ptr<ReadPlate::Response> response) {
        handle(request, response);
      });
    RCLCPP_INFO(get_logger(),
                "plate_reader ready: service plate_reader/read_plate");
  }

private:
  void handle(const std::shared_ptr<ReadPlate::Request> request,
              std::shared_ptr<ReadPlate::Response> response)
  {
    const std::string & req_path = request->image_path;
    if (req_path.empty()) {
      response->message = "image_path is empty";
      return;
    }
    fs::path resolved = resolve_path(req_path);
    if (resolved.empty()) {
      response->message = "file not found: " + req_path +
                          " (tried cwd, $PLATE_READER_IMAGE_DIR, "
                          "$HOME/plate_images, /tmp/plate_images)";
      RCLCPP_WARN(get_logger(), "%s", response->message.c_str());
      return;
    }

    cv::Mat img = cv::imread(resolved.string(), cv::IMREAD_COLOR);
    if (img.empty()) {
      response->message = "cannot decode image: " + resolved.string();
      RCLCPP_WARN(get_logger(), "%s", response->message.c_str());
      return;
    }

    PlateResult r;
    try {
      r = engine_->read_plate(img);
    } catch (const std::exception & e) {
      response->message = std::string("pipeline error: ") + e.what();
      RCLCPP_ERROR(get_logger(), "%s", response->message.c_str());
      return;
    }

    response->success = r.success;
    response->plate_number = r.plate_number;
    response->confidence = r.confidence;
    response->issuing_state = r.issuing_state;
    response->state_confidence = r.state_confidence;
    response->state_raw = r.state_raw;
    response->car_color = r.car_color;
    response->message = r.message;
    if (r.success) {
      RCLCPP_INFO(get_logger(),
                  "plate %s (conf %.2f) state '%s' (%.2f) color %s [%s]",
                  r.plate_number.c_str(), r.confidence,
                  r.issuing_state.c_str(), r.state_confidence,
                  r.car_color.c_str(), resolved.string().c_str());
    } else {
      RCLCPP_WARN(get_logger(), "no plate read from %s: %s",
                  resolved.string().c_str(), r.message.c_str());
    }
  }

  static fs::path resolve_path(const std::string & p)
  {
    std::vector<fs::path> roots{fs::path(".")};
    if (const char * env = std::getenv("PLATE_READER_IMAGE_DIR"))
      roots.push_back(fs::path(env));
    if (const char * home = std::getenv("HOME"))
      roots.push_back(fs::path(home) / "plate_images");
    roots.push_back(fs::path("/tmp/plate_images"));

    if (fs::path(p).is_absolute()) {
      if (fs::exists(p)) return fs::path(p);
      return {};
    }
    for (const auto & root : roots) {
      fs::path cand = root / p;
      std::error_code ec;
      if (fs::exists(cand, ec)) return cand;
    }
    return {};
  }

  std::unique_ptr<AlprEngine> engine_;
  rclcpp::Service<ReadPlate>::SharedPtr srv_;
};

}  // namespace plate_reader

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<plate_reader::PlateReaderNode>(
    rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}
