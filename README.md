# Plate Reader ~ offline license-plate reader ROS2 node
A ROS2 C++ service node for reading license plates. It takes the relative path of a `.png` showing the rear of a car, reads the license plate, and returns the plate number plus extra information and the car's color. The system is tuned for US-style plates.

Everything runs **offline on CPU** — OpenCV classical vision plus Tesseract OCR. No GPU and no internet.

## Limitations Warning
This package is NOT guaranteed to correctly read a car license plate! The plate reader OFTEN FAILS and returns wrong numbers and letters. Human validation is required ever time this package is used!


## Dependencies

System dependencies (implemented for Ubuntu 20.04):

```bash
sudo apt install ros-foxy-rclcpp ros-foxy-rosidl-default-generators \
                 libopencv-dev libtesseract-dev libleptonica-dev \
                 tesseract-ocr tesseract-ocr-eng
```

The node looks for `eng.traineddata` in: the `tessdata_path` parameter, then `TESSDATA_PREFIX`, then the system default (`apt install tesseract-ocr-eng` puts it in `/usr/share/tesseract-ocr/4.00/tessdata`).

### Porting to newer distros

On Jazzy+ replace the `rosidl_target_interfaces` line in CMakeLists.txt with `rosidl_get_typesupport_target` (see the comment in the file).


## Service interface

```
string image_path          # request
---
bool success               # a plate was found and read
string plate_number        # e.g. "ABC1234"
float64 confidence         # OCR confidence 0..1
string issuing_state       # currently not used
float64 state_confidence   # defaults to 0.0
string state_raw           # raw OCR of the plate's top strip (for review)
string car_color           # e.g. "silver"
string message             # "ok" or error description
```

Call it from the CLI:

```bash
ros2 service call /read_plate plate_reader/srv/ReadPlate \
  "{image_path: test_plate.png}"
```

## How the pipeline works

Each stage was validated against real 1280x720 ZED camera frames of a car rear — a close frame (plate 405x169, read **exactly**: ABC1234) and a far frame (plate only ~95x45 px, where the previous release failed outright with "OCR failed on all candidates"; now read as NZP9736 by the fusion stage below):

1. **Localization** — Sobel-x gradient (vertical strokes of plate characters) → Otsu → morphological close → contours. Same-row boxes are merged (plates split into text groups by wide spaces). Both the raw and merged boxes are kept as seeds. **Bright-rectangle seeds** are added: plates are retroreflective, so a large bright plate-shaped region is the plate itself — this recovers plates the text merges miss (e.g. wide internal spaces in "BLUE 777"). Each seed becomes several candidates: itself (if plate-shaped) plus 1x/1.6x/2.2x expansions to plate proportions (~2.4:1).
2. **Candidate refinement** — each candidate box is shrunk to the bright plate rectangle inside it. Localization seeds often include car body around the plate, which corrupts the warp proportions and the character band.
3. **Warp** — each candidate is perspective-corrected to a canonical plate **sized to the source** (~1.5x the quad width, clamped to 120..400 px). A fixed 400x200 canvas only interpolates (never adds) detail; a 75-px plate warped to 400 px is blurred beyond OCR repair — this was the root cause of the far-frame failure.
4. **Character band** — Otsu threshold, connected components; glyphs are row-clustered (the plate number row dominates the state banner by area); edge glyphs separated by background-scale gaps are dropped (only if also much smaller than the group median, so spaced formats keep their full-width groups); median vertical bounds avoid the plate frame.
5. **OCR** — two stages. Stage 1: bilateral filter (edge-preserving denoise), 5x LANCZOS upscale, Tesseract `PSM_SINGLE_LINE` with an A-Z0-9 whitelist. Stage 2 (only when stage 1 fails **and** the source quad is under ~250 px wide, i.e. glyphs are at the resolution limit): **fusion OCR** — x-projection glyph segmentation (column runs survive the low-resolution stroke fragmentation that breaks connected components), then three independent evidence sources per glyph: normalized cross-correlation against embedded font templates (two fonts, 36 glyphs each, generated into `templates.hpp`), whole-line Tesseract votes across preprocessing variants on a recomposed clean band, and per-glyph `PSM_SINGLE_CHAR` votes across upscale/prep variants; the sources are fused per position and decoded with a conservative US plate-format prior (LLLNNNN, NLLLNNN, ...). This recovers far plates single-shot OCR cannot read. Up to 12 candidates are OCR-verified.
6. **Selection** — reads are ranked: 5–8 characters strongly preferred (US plate formats), reads that fuzzy-match a US state name are demoted (that's the plate's banner, e.g. "NEW YORK").
7. **Extras** — issuing state from OCR of the plate's top strip matched against the 51 US states/territories by Levenshtein similarity (reported only at ratio ≥ 0.6, raw text always returned); car color from k-means dominant color of four body regions around the plate (bumper, trunk, left, right), brightness-weighted.

## Package layout

```
plate_reader/
├── CMakeLists.txt
├── package.xml
├── srv/ReadPlate.srv            # service definition
├── include/plate_reader/
│   ├── alpr.hpp                 # engine interface + shared helpers
│   └── templates.hpp            # embedded font glyph masks (generated)
├── src/
│   ├── alpr.cpp                 # the whole vision pipeline
│   ├── plate_reader_node.cpp    # ROS2 service node
│   ├── alpr_test_main.cpp       # standalone test harness (no ROS)
│   └── color_test_main.cpp      # color classifier unit test
└── launch/plate_reader.launch.xml
```

The two `*_test_main.cpp` files are not built by the ROS package (kept for reference/testing); to try the pipeline without ROS:

```bash
g++ -std=c++17 -O2 -I include $(pkg-config --cflags opencv4) \
    src/alpr.cpp src/alpr_test_main.cpp -o alpr_test \
    $(pkg-config --libs opencv4) -ltesseract
./alpr_test /usr/share/tesseract-ocr/4.00/tessdata some_car.png
```

`include/plate_reader/templates.hpp` is generated by `scripts/gen_templates.py` (renders A-Z0-9 in two bold fonts and emits the binary masks as a C++ array) — regenerate only if you change the font set.

A Python reference implementation of the identical pipeline (used to develop and validate the algorithm) is in `scripts/prototype_alpr_v2.py` with the fusion OCR in `scripts/fusion_ocr.py`.
