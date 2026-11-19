// plate_reader/alpr.hpp — offline US license-plate reader (OpenCV + Tesseract)
// C++ port of a pipeline validated against a real 1280x720 ZED camera frame
// (blurry plate read exactly: MZP9736) plus synthetic stress images.
//
// Stages:
//   1. Localization: Sobel-x edges -> close -> contours; same-row box merge;
//      seed expansion to plate proportions; multiple candidates ranked by
//      edge density.
//   2. Warp candidate to 400x200 canonical plate.
//   3. Character band: Otsu threshold, connected components, glyph row
//      clustering, edge-gap outlier removal, tight crop.
//   4. OCR: bilateral filter, LANCZOS 5x upscale, Tesseract psm 7 with
//      A-Z0-9 whitelist.
//   5. Result selection: prefer 5-8 char plates, penalize state banners.
//   6. Extras: state fuzzy match (edit distance) on the plate's top strip,
//      car color from k-means dominant color around the plate.

#ifndef PLATE_READER__ALPR_HPP_
#define PLATE_READER__ALPR_HPP_

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <map>

namespace plate_reader
{

struct PlateResult
{
  bool success = false;
  std::string plate_number;
  double confidence = 0.0;
  std::string issuing_state;
  double state_confidence = 0.0;
  std::string state_raw;
  std::string car_color;
  std::string message;
  cv::Rect bbox;         // plate bounding box in the source image
};

// 51 US states + territories (DC)
inline const std::vector<std::string> & us_states()
{
  static const std::vector<std::string> S = {
    "ALABAMA", "ALASKA", "ARIZONA", "ARKANSAS", "CALIFORNIA", "COLORADO",
    "CONNECTICUT", "DELAWARE", "FLORIDA", "GEORGIA", "HAWAII", "IDAHO",
    "ILLINOIS", "INDIANA", "IOWA", "KANSAS", "KENTUCKY", "LOUISIANA",
    "MAINE", "MARYLAND", "MASSACHUSETTS", "MICHIGAN", "MINNESOTA",
    "MISSISSIPPI", "MISSOURI", "MONTANA", "NEBRASKA", "NEVADA",
    "NEW HAMPSHIRE", "NEW JERSEY", "NEW MEXICO", "NEW YORK",
    "NORTH CAROLINA", "NORTH DAKOTA", "OHIO", "OKLAHOMA", "OREGON",
    "PENNSYLVANIA", "RHODE ISLAND", "SOUTH CAROLINA", "SOUTH DAKOTA",
    "TENNESSEE", "TEXAS", "UTAH", "VERMONT", "VIRGINIA", "WASHINGTON",
    "WEST VIRGINIA", "WISCONSIN", "WYOMING", "DISTRICT OF COLUMBIA"};
  return S;
}

// ------------------------------------------------------------- utilities
inline std::string to_upper(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return s;
}

inline std::string alnum_only(const std::string & s)
{
  std::string out;
  for (char c : s) {
    if ((c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) out += c;
  }
  return out;
}

inline std::string letters_spaces_only(const std::string & s)
{
  std::string out;
  bool last_space = true;
  for (char c : s) {
    if (c >= 'A' && c <= 'Z') { out += c; last_space = false; }
    else if (c == ' ' && !last_space) { out += ' '; last_space = true; }
  }
  while (!out.empty() && out.back() == ' ') out.pop_back();
  return out;
}

// Levenshtein-based similarity ratio (0..1), equivalent to Python's
// difflib.SequenceMatcher().ratio() for our purposes
inline double similarity(const std::string & a, const std::string & b)
{
  if (a.empty() && b.empty()) return 1.0;
  const size_t n = a.size(), m = b.size();
  std::vector<size_t> prev(m + 1), cur(m + 1);
  for (size_t j = 0; j <= m; ++j) prev[j] = j;
  for (size_t i = 1; i <= n; ++i) {
    cur[0] = i;
    for (size_t j = 1; j <= m; ++j) {
      size_t cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
      cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost});
    }
    std::swap(prev, cur);
  }
  size_t dist = prev[m];
  return 1.0 - static_cast<double>(dist) /
         static_cast<double>(n + m);
}

inline double median(std::vector<double> v)
{
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// ------------------------------------------------------------ car color
inline std::string classify_color(const cv::Vec3d & bgr)
{
  const double b = bgr[0], g = bgr[1], r = bgr[2];
  const double v = std::max({b, g, r});
  const double mn = std::min({b, g, r});
  const double s = (v > 1e-6) ? (v - mn) / v : 0.0;
  if (v < 45) return "black";
  if (s < 0.18) {
    if (v > 210) return "white";
    if (v > 120) return "silver";
    return "gray";
  }
  // hue-like rules (BGR ordering: blue=b, green=g, red=r)
  if (r >= g && g >= b && r > b) {          // red -> yellow sector
    if (g < 0.30 * r) return (v < 110) ? "brown" : "red";
    if (g < 0.60 * r) return (v < 110) ? "brown" : "orange";
    return (v < 110) ? "brown" : "yellow";
  }
  if (b > g && r > g) return (v > 150) ? "pink" : "magenta";
  if (g >= r && g >= b) return "green";
  if (b >= r && b >= g) {
    if (v < 90) return "navy";
    if (s > 0.4) return "blue";
    return "gray";
  }
  return "gray";
}

// ------------------------------------------------------- OCR interface
// Implemented in alpr.cpp against libtesseract; declared here so the
// pipeline class stays header-only for everything else.
struct OcrOutput
{
  std::string text;
  double confidence = 0.0;
};

class AlprEngine
{
public:
  // path to tessdata dir with eng.traineddata; empty = system default
  explicit AlprEngine(std::string tessdata_path = "");
  ~AlprEngine();
  AlprEngine(const AlprEngine &) = delete;
  AlprEngine & operator=(const AlprEngine &) = delete;

  // Main entry: read a plate from an image. Thread-safe (mutex around the
  // shared Tesseract handle).
  PlateResult read_plate(const cv::Mat & bgr);

private:
  struct Impl;
  Impl * pimpl_;
};

}  // namespace plate_reader

#endif  // PLATE_READER__ALPR_HPP_
