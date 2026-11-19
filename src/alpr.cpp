// plate_reader/alpr.cpp — implementation of the offline ALPR pipeline.
#include "plate_reader/alpr.hpp"
#include "plate_reader/templates.hpp"

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

#include <mutex>
#include <stdexcept>

namespace plate_reader
{

namespace
{

const char * ALNUM_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

// ------------------------------------------------------ localization
struct Seed
{
  int x, y, w, h;
  double edge_density;
};

std::vector<cv::Rect> find_plate_candidates(const cv::Mat & gray, int max_out = 16)
{
  const int H = gray.rows, W = gray.cols;
  cv::Mat sobel;
  cv::Sobel(gray, sobel, CV_8U, 1, 0, 3);
  cv::Mat sobel_t;
  cv::threshold(sobel, sobel_t, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
  cv::Mat close;
  cv::morphologyEx(sobel_t, close, cv::MORPH_CLOSE,
                   cv::getStructuringElement(cv::MORPH_RECT, cv::Size(23, 7)));
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(close, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

  auto box_list = std::vector<std::array<int, 4>>();
  std::vector<double> edge_dens;
  for (const auto & c : contours) {
    cv::Rect r = cv::boundingRect(c);
    if (r.width < 50 || r.height < 14 || r.height > H / 2) continue;
    cv::Mat roi = sobel(r);
    double ed = static_cast<double>(cv::countNonZero(roi > 80)) /
                static_cast<double>(r.area());
    box_list.push_back({{r.x, r.y, r.width, r.height}});
    edge_dens.push_back(ed);
  }

  // merge same-row boxes (vertical overlap >= 50% of the shorter, small gap)
  auto merged = box_list;
  auto merged_ed = edge_dens;
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < merged.size() && !changed; ++i) {
      for (size_t j = i + 1; j < merged.size() && !changed; ++j) {
        auto & a = merged[i];
        auto & b = merged[j];
        int vy = std::min(a[1] + a[3], b[1] + b[3]) - std::max(a[1], b[1]);
        if (vy < 0.5 * std::min(a[3], b[3])) continue;
        int gap = std::max({b[0] - (a[0] + a[2]), a[0] - (b[0] + b[2]), 0});
        if (gap > 1.2 * std::max(a[3], b[3])) continue;
        int nx = std::min(a[0], b[0]), ny = std::min(a[1], b[1]);
        int nx2 = std::max(a[0] + a[2], b[0] + b[2]);
        int ny2 = std::max(a[1] + a[3], b[1] + b[3]);
        merged[i] = {{nx, ny, nx2 - nx, ny2 - ny}};
        merged_ed[i] = std::max(merged_ed[i], merged_ed[j]);
        merged.erase(merged.begin() + j);
        merged_ed.erase(merged_ed.begin() + j);
        changed = true;
      }
    }
  }

  // seeds = raw + merged
  std::vector<Seed> seeds;
  for (size_t k = 0; k < box_list.size(); ++k) {
    auto & b = box_list[k];
    double ar = double(b[2]) / double(b[3]);
    if (ar >= 1.5 && ar <= 8.0 && b[2] >= 80 && edge_dens[k] >= 0.06)
      seeds.push_back({b[0], b[1], b[2], b[3], edge_dens[k]});
  }
  for (size_t k = 0; k < merged.size(); ++k) {
    auto & b = merged[k];
    double ar = double(b[2]) / double(b[3]);
    if (ar >= 1.5 && ar <= 8.0 && b[2] >= 80 && merged_ed[k] >= 0.06)
      seeds.push_back({b[0], b[1], b[2], b[3], merged_ed[k]});
  }

  // bright-rectangle seeds: plates are retroreflective, so the bright
  // plate-shaped region IS the plate; text merging can miss plates split
  // by wide internal character spaces (e.g. "BLUE 777")
  std::vector<std::array<int, 4>> bright_seeds;
  {
    cv::Mat bt;
    cv::threshold(gray, bt, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Mat btc;
    cv::morphologyEx(bt, btc, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 3)));
    std::vector<std::vector<cv::Point>> bcnts;
    cv::findContours(btc, bcnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto & c : bcnts) {
      cv::Rect r = cv::boundingRect(c);
      double ar = double(r.width) / double(r.height);
      if (r.width < 0.12 * W || ar < 1.6 || ar > 4.5) continue;
      if (cv::contourArea(c) < 0.35 * r.area()) continue;
      bright_seeds.push_back({{r.x, r.y, r.width, r.height}});
      seeds.push_back({r.x, r.y, r.width, r.height, 0.05});
    }
  }

  // expand each seed into up to 4 candidates: itself (if plate-shaped)
  // plus 1x/1.6x/2.2x-width expansions to plate proportions
  std::vector<std::pair<double, cv::Rect>> out;
  auto is_bright = [&](const std::array<int, 4> & b) {
    for (auto & s : bright_seeds)
      if (s == b) return true;
    return false;
  };
  for (const auto & s : seeds) {
    double ar = double(s.w) / double(s.h);
    double sc = 1.0 + 2.0 * std::min(s.edge_density, 0.4);
    // bright seeds already passed a strict shape+fill test, so they get a
    // lower edge-density bar (their box includes the empty plate margin)
    double ed_bar = is_bright({{s.x, s.y, s.w, s.h}}) ? 0.04 : 0.06;
    if (s.edge_density < ed_bar) continue;
    if (ar >= 1.8 && ar <= 5.0)
      out.push_back({sc, cv::Rect(s.x, s.y, s.w, s.h)});
    for (double f : {1.0, 1.6, 2.2}) {
      int ph = int(std::max(s.h * 1.9, s.w * f / 2.4));
      int pw = int(std::max(double(s.w) * f, ph * 2.4));
      if (pw > 1.5 * W || ph > H) continue;
      double cx = s.x + s.w / 2.0, cy = s.y + s.h / 2.0;
      int py0 = int(cy - ph * 0.42);
      int px0 = int(cx - pw / 2.0);
      int bx = std::max(0, std::min(px0, W - pw));
      int by = std::max(0, std::min(py0, H - ph));
      out.push_back({sc, cv::Rect(bx, by, pw, ph)});
    }
  }
  std::sort(out.begin(), out.end(),
            [](const auto & a, const auto & b) { return a.first > b.first; });
  if (out.size() > size_t(max_out)) out.resize(max_out);

  std::vector<cv::Rect> rects;
  for (auto & p : out) rects.push_back(p.second);
  return rects;
}

// Shrink a candidate box to the bright plate rectangle inside it (plates
// are retroreflective); localization seeds often include car body around
// the plate, which corrupts the warp proportions and the character band.
cv::Rect refine_candidate(const cv::Mat & bgr, const cv::Rect & box)
{
  const int H = bgr.rows, W = bgr.cols;
  cv::Rect r = box & cv::Rect(0, 0, W, H);
  if (r.width < 10 || r.height < 10) return box;
  cv::Mat roi;
  cv::cvtColor(bgr(r), roi, cv::COLOR_BGR2GRAY);
  cv::Mat th;
  cv::threshold(roi, th, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
  cv::Mat thc;
  cv::morphologyEx(th, thc, cv::MORPH_CLOSE,
                   cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 3)));
  std::vector<std::vector<cv::Point>> cnts;
  cv::findContours(thc, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  cv::Rect best;
  for (const auto & c : cnts) {
    cv::Rect b = cv::boundingRect(c);
    if (b.width < 0.35 * r.width || b.height < 0.25 * r.height) continue;
    double ar = double(b.width) / double(b.height);
    if (ar < 1.5 || ar > 4.5) continue;
    if (cv::contourArea(c) < 0.35 * b.area()) continue;
    if (b.area() > best.area()) best = b;
  }
  if (best.area() == 0) return box;
  return cv::Rect(r.x + best.x, r.y + best.y, best.width, best.height);
}

// ------------------------------------------------------------- warp
// Perspective-correct to a canonical plate SIZED TO THE SOURCE: the output
// resolution tracks the quad width (~1.5x, clamped to 120..400 px).
// Warping a 75-px plate to a fixed 400x200 canvas only interpolates
// (never adds) detail and blurs glyphs beyond OCR repair.
cv::Mat warp_plate(const cv::Mat & img, const cv::Rect & r)
{
  double src_w = r.width;
  int out_w = int(std::min(400.0, std::max(120.0, 1.5 * src_w)));
  int out_h = out_w / 2;
  cv::Mat src_quad = (cv::Mat_<float>(4, 2) <<
                      r.x, r.y, r.x + r.width, r.y,
                      r.x + r.width, r.y + r.height, r.x, r.y + r.height);
  cv::Mat dst_quad = (cv::Mat_<float>(4, 2) <<
                      0, 0, out_w - 1.0f, 0,
                      out_w - 1.0f, out_h - 1.0f, 0, out_h - 1.0f);
  cv::Mat M = cv::getPerspectiveTransform(src_quad, dst_quad);
  cv::Mat warped;
  cv::warpPerspective(img, warped, M, cv::Size(out_w, out_h));
  return warped;
}

// -------------------------------------------------------- char band
struct Glyph
{
  int x, y, w, h;
};

cv::Mat char_band(const cv::Mat & warped)
{
  cv::Mat g;
  cv::cvtColor(warped, g, cv::COLOR_BGR2GRAY);
  cv::Mat th;
  cv::threshold(g, th, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
  cv::Mat opened;
  cv::morphologyEx(th, opened, cv::MORPH_OPEN,
                   cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2)));
  int H = opened.rows, W = opened.cols;
  cv::Mat labels, stats, cents;
  int n = cv::connectedComponentsWithStats(opened, labels, stats, cents, 8);
  std::vector<Glyph> glyphs;
  for (int i = 1; i < n; ++i) {
    int x = stats.at<int>(i, cv::CC_STAT_LEFT);
    int y = stats.at<int>(i, cv::CC_STAT_TOP);
    int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
    int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
    int a = stats.at<int>(i, cv::CC_STAT_AREA);
    if (a < W * H * 0.0005 || h < H * 0.15 || h > H * 0.7 || w > W * 0.3)
      continue;
    double cy = y + h / 2.0;
    if (!(H * 0.15 < cy && cy < H * 0.95)) continue;
    glyphs.push_back({x, y, w, h});
  }
  if (glyphs.size() < 4) return cv::Mat();
  std::sort(glyphs.begin(), glyphs.end(),
            [](const Glyph & a, const Glyph & b) { return a.x < b.x; });

  // cluster glyphs into rows by vertical center; keep the row with the
  // largest total glyph area (the plate number dominates the state banner)
  struct Row { double cy_sum = 0; double area = 0; int cnt = 0; std::vector<int> idx; };
  std::vector<Row> rows;
  for (size_t i = 0; i < glyphs.size(); ++i) {
    double cy = glyphs[i].y + glyphs[i].h / 2.0;
    double area = glyphs[i].w * glyphs[i].h;
    bool placed = false;
    for (auto & row : rows) {
      double mean_cy = row.cy_sum / row.cnt;
      if (std::fabs(mean_cy - cy) < 0.35 * glyphs[i].h) {
        row.cy_sum += cy; row.area += area; row.cnt++;
        row.idx.push_back(int(i));
        placed = true;
        break;
      }
    }
    if (!placed) rows.push_back({cy, area, 1, {int(i)}});
  }
  Row best = *std::max_element(rows.begin(), rows.end(),
    [](const Row & a, const Row & b) { return a.area < b.area; });
  std::vector<Glyph> row_glyphs;
  for (int i : best.idx) row_glyphs.push_back(glyphs[i]);
  if (row_glyphs.size() < 4) return cv::Mat();
  std::sort(row_glyphs.begin(), row_glyphs.end(),
            [](const Glyph & a, const Glyph & b) { return a.x < b.x; });

  // drop leading/trailing glyphs separated by a gap much wider than the
  // widest plausible internal space; a wide gap only disqualifies an edge
  // glyph that is ALSO much smaller than the group median (spaced plate
  // formats like "BLUE 777" keep full-width groups intact)
  if (row_glyphs.size() > 4) {
    std::vector<double> areas;
    for (auto & g : row_glyphs) areas.push_back(double(g.w) * g.h);
    double med_area = median(areas);
    std::vector<double> gaps;
    for (size_t i = 0; i + 1 < row_glyphs.size(); ++i)
      gaps.push_back(row_glyphs[i + 1].x - (row_glyphs[i].x + row_glyphs[i].w));
    std::vector<double> sg = gaps;
    std::sort(sg.begin(), sg.end());
    double med_gap = sg[sg.size() / 2];
    double edge_limit = std::max(std::min(8.0 * med_gap, 0.25 * W), 30.0);
    while (row_glyphs.size() > 4) {
      double lg = row_glyphs[1].x - (row_glyphs[0].x + row_glyphs[0].w);
      double tg = row_glyphs[row_glyphs.size() - 1].x -
                  (row_glyphs[row_glyphs.size() - 2].x +
                   row_glyphs[row_glyphs.size() - 2].w);
      bool lead_small = double(row_glyphs.front().w) * row_glyphs.front().h <
                        0.35 * med_area;
      bool tail_small = double(row_glyphs.back().w) * row_glyphs.back().h <
                        0.35 * med_area;
      if (lg > edge_limit && lead_small) row_glyphs.erase(row_glyphs.begin());
      else if (tg > edge_limit && tail_small) row_glyphs.pop_back();
      else break;
    }
  }

  int x0 = std::max(0, row_glyphs.front().x - 6);
  int x1 = std::min(W, row_glyphs.back().x + row_glyphs.back().w + 6);
  std::vector<double> tops, bots;
  for (auto & g : row_glyphs) { tops.push_back(g.y); bots.push_back(g.y + g.h); }
  int y0 = std::max(0, int(median(tops)) - 6);
  int y1 = std::min(H, int(median(bots)) + 6);
  if (x1 - x0 < 0.3 * W || y1 - y0 < 8) return cv::Mat();
  return warped(cv::Rect(x0, y0, x1 - x0, y1 - y0)).clone();
}

// ----------------------------------------------------- fusion OCR
// X-projection segmentation: column runs of foreground pixels. At low
// resolution connected components fragment vertically, but per-character
// column runs stay intact. Narrow runs (<3 px) merge into an adjacent run.
std::vector<Glyph> segment_glyphs(const cv::Mat & band_gray, cv::Mat & th_out)
{
  cv::Mat th;
  cv::threshold(band_gray, th, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
  cv::Mat opened;
  cv::morphologyEx(th, opened, cv::MORPH_OPEN,
                   cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2)));
  th_out = opened;
  const int H = opened.rows;
  // column ink profile
  std::vector<int> proj(opened.cols, 0);
  for (int x = 0; x < opened.cols; ++x) {
    int cnt = 0;
    for (int y = 0; y < H; ++y) cnt += (opened.at<uint8_t>(y, x) > 0);
    proj[x] = cnt;
  }
  std::vector<std::array<int, 2>> runs;
  bool in_run = false;
  int start = 0;
  for (int x = 0; x < opened.cols; ++x) {
    if (proj[x] > 0 && !in_run) { start = x; in_run = true; }
    else if (proj[x] == 0 && in_run) { runs.push_back({{start, x}}); in_run = false; }
  }
  if (in_run) runs.push_back({{start, opened.cols}});
  // merge narrow fragments
  std::vector<std::array<int, 2>> merged;
  for (auto & r : runs) {
    if (!merged.empty() &&
        (r[1] - r[0] < 3 ||
         merged.back()[1] - merged.back()[0] < 3) &&
        r[0] - merged.back()[1] <= 2)
      merged.back()[1] = r[1];
    else
      merged.push_back(r);
  }
  std::vector<Glyph> glyphs;
  for (auto & m : merged) {
    int y0 = H, y1 = 0;
    for (int y = 0; y < H; ++y) {
      bool any = false;
      for (int x = m[0]; x < m[1]; ++x)
        if (opened.at<uint8_t>(y, x) > 0) { any = true; break; }
      if (any) { y0 = std::min(y0, y); y1 = std::max(y1, y + 1); }
    }
    int h = y1 - y0;
    if (h < 0.15 * H || h > 0.95 * H) continue;
    glyphs.push_back({m[0], y0, m[1] - m[0], h});
  }
  return glyphs;
}

double ncc(const cv::Mat & a, const cv::Mat & b)
{
  cv::Mat A, B;
  a.convertTo(A, CV_32F);
  b.convertTo(B, CV_32F);
  A -= cv::mean(A)[0];
  B -= cv::mean(B)[0];
  double da = cv::norm(A), db = cv::norm(B);
  if (da < 1e-6 || db < 1e-6) return 0.0;
  return A.dot(B) / (da * db);
}

// per-glyph template NCC scores (max over the two embedded fonts)
void template_scores(const cv::Mat & glyph_mask, std::array<double, 36> & scores)
{
  const auto & tmpls = glyph_templates();
  const std::string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  for (int f = 0; f < 2; ++f) {
    for (int ci = 0; ci < 36; ++ci) {
      const GlyphTemplate & t = tmpls[f * 36 + ci];
      cv::Mat tt(t.h, t.w, CV_8U);
      for (int y = 0; y < t.h; ++y)
        for (int x = 0; x < t.w; ++x)
          tt.at<uint8_t>(y, x) = t.mask[y * t.w + x] ? 255 : 0;
      cv::Mat resized;
      cv::resize(tt, resized, glyph_mask.size(), 0, 0, cv::INTER_AREA);
      cv::Mat rf;
      resized.convertTo(rf, CV_32F, 1.0 / 255.0);
      double s = ncc(rf, glyph_mask);
      if (s > scores[ci]) scores[ci] = s;
    }
  }
}

// whole-line tesseract reads across preprocessing variants
std::vector<std::string> tess_line_reads(tesseract::TessBaseAPI & api,
                                         const cv::Mat & band_gray)
{
  std::vector<std::string> reads;
  api.SetVariable("tessedit_char_whitelist", ALNUM_WHITELIST);
  for (double scale : {5.0, 8.0}) {
    cv::Mat up;
    cv::resize(band_gray, up, cv::Size(), scale, scale, cv::INTER_LANCZOS4);
    std::vector<cv::Mat> variants;
    variants.push_back(up);
    cv::Mat otsu;
    cv::threshold(up, otsu, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    variants.push_back(otsu);
    cv::Mat blur;
    cv::GaussianBlur(up, blur, cv::Size(0, 0), 3.0);
    cv::Mat sharp;
    cv::addWeighted(up, 2.0, blur, -1.0, 0, sharp);
    variants.push_back(sharp);
    for (int psm : {7, 6}) {
      for (auto & v : variants) {
        cv::Mat padded;
        cv::copyMakeBorder(v, padded, 30, 30, 60, 60, cv::BORDER_REPLICATE);
        api.SetPageSegMode(static_cast<tesseract::PageSegMode>(psm));
        api.SetImage(padded.data, padded.cols, padded.rows, 1, padded.step);
        char * txt = api.GetUTF8Text();
        std::string t = alnum_only(to_upper(txt ? txt : ""));
        delete[] txt;
        if (t.size() >= 4) reads.push_back(t);
      }
    }
  }
  return reads;
}

// per-glyph single-char OCR (psm 10) across upscale/prep variants
std::vector<std::array<int, 36>> p10_ensemble(tesseract::TessBaseAPI & api,
                                              const cv::Mat & th,
                                              const std::vector<Glyph> & glyphs)
{
  std::vector<std::array<int, 36>> votes(glyphs.size());
  for (auto & v : votes) v.fill(0);
  const std::string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  api.SetVariable("tessedit_char_whitelist", ALNUM_WHITELIST);
  for (size_t gi = 0; gi < glyphs.size(); ++gi) {
    const Glyph & g = glyphs[gi];
    int y0 = std::max(0, g.y - 2), y1 = std::min(th.rows, g.y + g.h + 2);
    int x0 = std::max(0, g.x - 2), x1 = std::min(th.cols, g.x + g.w + 2);
    if (y1 - y0 < 2 || x1 - x0 < 2) continue;
    cv::Mat inv;
    cv::bitwise_not(th(cv::Rect(x0, y0, x1 - x0, y1 - y0)), inv);
    for (int target_h : {32, 48, 64}) {
      double f = double(target_h) / std::max(1, g.h);
      cv::Mat big;
      cv::resize(inv, big, cv::Size(), f, f, cv::INTER_LANCZOS4);
      std::vector<cv::Mat> preps;
      preps.push_back(big);
      cv::Mat bil;
      cv::bilateralFilter(big, bil, 9, 60, 60);
      preps.push_back(bil);
      cv::Mat otsu;
      cv::threshold(big, otsu, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
      preps.push_back(otsu);
      for (auto & p : preps) {
        cv::Mat padded;
        cv::copyMakeBorder(p, padded, 25, 25, 25, 25, cv::BORDER_CONSTANT,
                           cv::Scalar(255));
        api.SetPageSegMode(tesseract::PSM_SINGLE_CHAR);
        api.SetImage(padded.data, padded.cols, padded.rows, 1, padded.step);
        char * txt = api.GetUTF8Text();
        std::string t = alnum_only(to_upper(txt ? txt : ""));
        delete[] txt;
        if (t.size() == 1) {
          size_t ci = chars.find(t[0]);
          if (ci != std::string::npos) votes[gi][ci]++;
        }
      }
    }
  }
  return votes;
}

// common US plate patterns by length (L=letter, N=digit) — conservative,
// formats actually issued by US states
const std::map<int, std::vector<std::string>> & plate_patterns()
{
  static const std::map<int, std::vector<std::string>> P = {
    {5, {"LLNNN", "LNNLL", "NNLLL", "LLLNN", "NLLLL", "LNNNN", "NNNNL"}},
    {6, {"LLLNNN", "NNNLLL", "LLNNNN", "NNNNLL", "LNNNNN", "NNLLLL"}},
    {7, {"LLLNNNN", "NLLLNNN", "LNNNNNN", "NNNLLLL", "NNNNLLL", "LLNNNNN",
         "NNLLLLN", "NNLLLLL"}},
    {8, {"LLNNNNNN", "NNNNNLLL", "LLLNNNNN", "NNNLLLLN", "NNLLLLNN",
         "LNNLLNNN"}},
  };
  return P;
}

bool is_letter(char c) { return c >= 'A' && c <= 'Z'; }
bool is_digit(char c) { return c >= '0' && c <= '9'; }

bool matches_pattern(const std::string & t)
{
  auto it = plate_patterns().find(int(t.size()));
  if (it == plate_patterns().end()) return false;
  for (const auto & p : it->second) {
    bool ok = true;
    for (size_t i = 0; i < t.size(); ++i) {
      bool want_letter = (p[i] == 'L');
      if (is_letter(t[i]) != want_letter) { ok = false; break; }
    }
    if (ok) return true;
  }
  return false;
}

// Triple-source fusion decode: template NCC + whole-line votes + per-glyph
// psm10 votes, then a US format prior. Recovers far/small plates whose
// glyphs are at the resolution limit, where single-shot tesseract fails.
struct FusionOut
{
  std::string text;
  double conf = 0.0;
};

FusionOut fusion_read(tesseract::TessBaseAPI & api, cv::Mat band_gray)
{
  FusionOut out;
  // supersample tiny bands before thresholding: smoother glyph masks
  if (band_gray.rows < 40) {
    cv::Mat up;
    cv::resize(band_gray, up, cv::Size(), 4.0, 4.0, cv::INTER_LANCZOS4);
    band_gray = up;
  }
  cv::Mat th;
  std::vector<Glyph> glyphs = segment_glyphs(band_gray, th);
  if (glyphs.size() < 4 || glyphs.size() > 8) return out;
  const size_t n = glyphs.size();
  const std::string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

  // source 1: template NCC per glyph
  std::vector<std::array<double, 36>> t_scores(n);
  for (size_t i = 0; i < n; ++i) {
    t_scores[i].fill(0.0);
    const Glyph & g = glyphs[i];
    cv::Mat mask(g.h, g.w, CV_32F);
    for (int y = 0; y < g.h; ++y)
      for (int x = 0; x < g.w; ++x)
        mask.at<float>(y, x) =
          (th.at<uint8_t>(g.y + y, g.x + x) > 0) ? 1.0f : 0.0f;
    template_scores(mask, t_scores[i]);
  }

  // source 2: whole-line reads on a recomposed clean band — the raw band
  // often contains border/frame fragments that shift OCR's character count
  cv::Mat canvas(th.rows + 8, th.cols + 8, CV_8U, cv::Scalar(255));
  for (const auto & g : glyphs) {
    for (int y = 0; y < g.h; ++y)
      for (int x = 0; x < g.w; ++x)
        canvas.at<uint8_t>(g.y + 4 + y, g.x + 4 + x) =
          255 - th.at<uint8_t>(g.y + y, g.x + x);
  }
  std::vector<std::string> reads = tess_line_reads(api, canvas);
  std::vector<std::array<int, 36>> votes(n);
  for (auto & v : votes) v.fill(0);
  for (const auto & r : reads) {
    if (r.size() != n) continue;
    for (size_t i = 0; i < n; ++i) {
      size_t ci = chars.find(r[i]);
      if (ci != std::string::npos) votes[i][ci]++;
    }
  }

  // source 3: per-glyph psm10 ensemble
  std::vector<std::array<int, 36>> p10 = p10_ensemble(api, th, glyphs);

  // combine: normalize each source to max=1 per position, equal weights
  std::vector<std::array<double, 36>> comb(n);
  for (size_t i = 0; i < n; ++i) {
    double tmax = 0, vmax = 0, pmax = 0;
    for (int c = 0; c < 36; ++c) {
      tmax = std::max(tmax, t_scores[i][c]);
      vmax = std::max(vmax, double(votes[i][c]));
      pmax = std::max(pmax, double(p10[i][c]));
    }
    tmax += 1e-6;
    for (int c = 0; c < 36; ++c)
      comb[i][c] = std::max(t_scores[i][c], 0.0) / tmax +
                   (vmax ? votes[i][c] / vmax : 0.0) +
                   (pmax ? p10[i][c] / pmax : 0.0);
  }

  // unconstrained decode
  std::string uncon;
  double uncon_s = 0;
  for (size_t i = 0; i < n; ++i) {
    int bi = 0;
    for (int c = 1; c < 36; ++c) if (comb[i][c] > comb[i][bi]) bi = c;
    uncon += chars[bi];
    uncon_s += comb[i][bi];
  }

  // format prior: if the unconstrained text matches a known US pattern keep
  // it; otherwise take the best pattern-constrained decode (max over a
  // subset can never beat the unconstrained argmax, so compare patterns only)
  std::string best_txt = uncon;
  double best_s = uncon_s;
  if (!matches_pattern(uncon)) {
    auto it = plate_patterns().find(int(n));
    if (it != plate_patterns().end()) {
      best_s = -1.0;
      for (const auto & p : it->second) {
        double s = 0;
        std::string txt;
        for (size_t i = 0; i < n; ++i) {
          int bi = -1;
          for (int c = 0; c < 36; ++c) {
            bool want_letter = (p[i] == 'L');
            bool is_l = is_letter(chars[c]);
            if (is_l != want_letter) continue;
            if (bi < 0 || comb[i][c] > comb[i][bi]) bi = c;
          }
          if (bi < 0) bi = 0;
          s += comb[i][bi];
          txt += chars[bi];
        }
        if (s > best_s) { best_s = s; best_txt = txt; }
      }
    }
  }
  out.text = best_txt;
  out.conf = std::min(1.0, best_s / double(n) / 3.0);
  return out;
}

// ------------------------------------------------------------ state OCR
struct StateGuess
{
  std::string state;
  double ratio = 0.0;
  std::string raw;
};

StateGuess ocr_state(tesseract::TessBaseAPI & api, const cv::Mat & warped)
{
  StateGuess best;
  const int H = warped.rows;
  for (double frac : {0.20, 0.25}) {
    cv::Mat strip = warped(cv::Rect(0, 0, warped.cols, int(H * frac))).clone();
    cv::Mat g;
    cv::cvtColor(strip, g, cv::COLOR_BGR2GRAY);
    cv::Mat bil;
    cv::bilateralFilter(g, bil, 9, 60, 60);
    cv::Mat up;
    cv::resize(bil, up, cv::Size(), 4.0, 4.0, cv::INTER_LANCZOS4);
    for (int psm : {7, 6}) {
      api.SetPageSegMode(static_cast<tesseract::PageSegMode>(psm));
      // clear the plate whitelist so spaces/letters OCR properly here
      api.SetVariable("tessedit_char_whitelist", "");
      api.SetImage(up.data, up.cols, up.rows, 1, up.step);
      char * txt = api.GetUTF8Text();
      std::string t = to_upper(txt ? txt : "");
      delete[] txt;
      t = letters_spaces_only(t);
      if (t.empty()) continue;
      for (const auto & st : us_states()) {
        double r = similarity(t, st);
        if (r > best.ratio) best = {st, r, t};
      }
    }
  }
  return best;
}

// ------------------------------------------------------------- color
std::string car_color(const cv::Mat & img, const cv::Rect & plate)
{
  const int H = img.rows, W = img.cols;
  int cx = plate.x + plate.width / 2;
  auto dom = [&](cv::Rect r) -> cv::Vec3d {
    r &= cv::Rect(0, 0, W, H);
    if (r.width < 20 || r.height < 20) return cv::Vec3d(-1, -1, -1);
    cv::Mat roi = img(r).clone().reshape(1, r.area());
    cv::Mat f;
    roi.convertTo(f, CV_32F);
    cv::Mat labels, centers;
    int k = 3;
    if (f.rows < 3 * k) return cv::Vec3d(-1, -1, -1);
    cv::kmeans(f, k, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                                10, 1.0),
               3, cv::KMEANS_PP_CENTERS, centers);
    // dominant cluster
    std::vector<int> counts(k, 0);
    for (int i = 0; i < labels.rows; ++i) counts[labels.at<int>(i, 0)]++;
    int bi = int(std::max_element(counts.begin(), counts.end()) - counts.begin());
    return cv::Vec3d(centers.at<float>(bi, 0), centers.at<float>(bi, 1),
                     centers.at<float>(bi, 2));
  };
  std::vector<cv::Vec3d> regions = {
    dom(cv::Rect(cx - 160, plate.y + plate.height + 15, 320, 105)),  // bumper
    dom(cv::Rect(cx - 120, plate.y - 70, 240, 62)),                  // trunk
    dom(cv::Rect(plate.x - plate.width - 40, plate.y - 40,
                 plate.width, plate.height + 80)),                    // left
    dom(cv::Rect(plate.x + plate.width + 20, plate.y - 40,
                 plate.width, plate.height + 80)),                    // right
  };
  std::vector<cv::Vec3d> valid;
  for (auto & r : regions) if (r[0] >= 0) valid.push_back(r);
  if (valid.empty()) return "unknown";
  // brightness^2 weighted blend: well-lit panels dominate
  cv::Vec3d blend(0, 0, 0);
  double wsum = 0;
  for (auto & r : valid) {
    double v = std::max((r[0] + r[1] + r[2]) / 3.0, 1.0);
    double w = v * v;
    blend += w * r;
    wsum += w;
  }
  blend *= 1.0 / wsum;
  return classify_color(blend);
}

bool looks_like_plate(const std::string & t)
{
  size_t n = t.size();
  return n >= 4 && n <= 8 &&
         t.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") ==
           std::string::npos;
}

}  // namespace

// -------------------------------------------------------- engine impl
struct AlprEngine::Impl
{
  tesseract::TessBaseAPI api;
  std::mutex mtx;
};

AlprEngine::AlprEngine(std::string tessdata_path)
: pimpl_(new Impl)
{
  const char * path = tessdata_path.empty() ? nullptr : tessdata_path.c_str();
  if (pimpl_->api.Init(path, "eng") != 0) {
    delete pimpl_;
    throw std::runtime_error("failed to init tesseract (check tessdata)");
  }
}

AlprEngine::~AlprEngine()
{
  pimpl_->api.End();
  delete pimpl_;
}

PlateResult AlprEngine::read_plate(const cv::Mat & bgr)
{
  PlateResult res;
  if (bgr.empty()) {
    res.message = "empty image";
    return res;
  }
  std::lock_guard<std::mutex> lock(pimpl_->mtx);

  cv::Mat gray;
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  std::vector<cv::Rect> cands = find_plate_candidates(gray);
  if (cands.empty()) {
    res.message = "no plate candidate found";
    return res;
  }

  struct Hit
  {
    std::string text;
    double conf;
    cv::Rect bbox;
    cv::Mat warped;
  };
  std::vector<Hit> hits;

  for (const auto & raw_r : cands) {
    cv::Rect r = refine_candidate(bgr, raw_r);
    cv::Mat warped = warp_plate(bgr, r);
    cv::Mat band = char_band(warped);
    if (band.empty()) continue;

    // stage 1 OCR: bilateral + LANCZOS 5x, psm 7, A-Z0-9 whitelist
    cv::Mat g;
    cv::cvtColor(band, g, cv::COLOR_BGR2GRAY);
    cv::Mat bil;
    cv::bilateralFilter(g, bil, 9, 60, 60);
    cv::Mat up;
    cv::resize(bil, up, cv::Size(), 5.0, 5.0, cv::INTER_LANCZOS4);
    pimpl_->api.SetPageSegMode(tesseract::PSM_SINGLE_LINE);
    pimpl_->api.SetVariable("tessedit_char_whitelist", ALNUM_WHITELIST);
    pimpl_->api.SetImage(up.data, up.cols, up.rows, 1, up.step);
    char * txt = pimpl_->api.GetUTF8Text();
    std::string t = alnum_only(to_upper(txt ? txt : ""));
    delete[] txt;
    double conf = pimpl_->api.MeanTextConf() / 100.0;

    // stage 2: fusion OCR fallback — only for far/small plates (the source
    // quad under ~250 px wide, i.e. glyphs near the resolution limit), where
    // single-shot tesseract fails; on close plates it only burns time
    if (!looks_like_plate(t) && r.width < 250) {
      FusionOut f = fusion_read(pimpl_->api, g);
      if (looks_like_plate(f.text)) {
        t = f.text;
        conf = f.conf;
      }
    }
    if (looks_like_plate(t))
      hits.push_back({t, conf, r, warped});
  }

  if (hits.empty()) {
    res.message = "OCR failed on all candidates";
    return res;
  }

  // rank: prefer 5-8 chars; penalize state banners
  auto quality = [](const Hit & h) {
    double fmt = (h.text.size() >= 5 && h.text.size() <= 8) ? 2.0 : 0.25;
    double st = 0.0;
    for (const auto & s : us_states()) st = std::max(st, similarity(h.text, s));
    double banner = (st > 0.75) ? 0.1 : 1.0;
    return fmt * banner * (0.5 + 0.5 * h.conf);
  };
  std::sort(hits.begin(), hits.end(), [&](const Hit & a, const Hit & b) {
    double qa = quality(a), qb = quality(b);
    if (std::fabs(qa - qb) > 1e-9) return qa > qb;
    // tie-break 1: bbox aspect ratio closer to a real plate (~2.4)
    double ar_a = double(a.bbox.width) / a.bbox.height;
    double ar_b = double(b.bbox.width) / b.bbox.height;
    double da = std::fabs(ar_a - 2.4), db = std::fabs(ar_b - 2.4);
    if (std::fabs(da - db) > 1e-9) return da < db;
    // tie-break 2: tighter box => better color sampling
    return a.bbox.area() < b.bbox.area();
  });
  const Hit & best = hits.front();

  res.success = true;
  res.plate_number = best.text;
  res.confidence = best.conf;
  res.bbox = best.bbox;

  StateGuess sg = ocr_state(pimpl_->api, best.warped);
  if (sg.ratio >= 0.6) res.issuing_state = sg.state;
  res.state_confidence = sg.ratio;
  res.state_raw = sg.raw;
  res.car_color = car_color(bgr, best.bbox);
  res.message = "ok";
  return res;
}

}  // namespace plate_reader
