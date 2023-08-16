#include <cmath>

#include "mediapipe/calculators/util/detections_to_rects_calculator.h"
#include "mediapipe/calculators/util/detections_to_rects_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

namespace {}  // namespace

// Generates a hand ROI based on a hand detection derived from hand-related pose
// landmarks.
//
// Inputs:
//   DETECTION - Detection.
//     Detection to convert to ROI. Must contain 3 key points indicating: wrist,
//     pinky and index fingers.
//
//   IMAGE_SIZE - std::pair<int, int>
//     Image width and height.
//
// Outputs:
//   NORM_RECT - NormalizedRect.
//     ROI based on passed input.
//
// Examples
// node {
//   calculator: "HandDetectionsFromPoseToRectsCalculator"
//   input_stream: "DETECTION:hand_detection_from_pose"
//   input_stream: "IMAGE_SIZE:image_size"
//   output_stream: "NORM_RECT:hand_roi_from_pose"
// }
class HandDetectionsFromPoseToRectsCalculator
    : public DetectionsToRectsCalculator {
 public:
  absl::Status Open(CalculatorContext* cc) override;

 private:
  ::absl::Status DetectionToNormalizedRect(const Detection& detection,
                                           const DetectionSpec& detection_spec,
                                           NormalizedRect* rect) override;
  absl::Status ComputeRotation(const Detection& detection,
                               const DetectionSpec& detection_spec,
                               float* rotation) override;
};
REGISTER_CALCULATOR(HandDetectionsFromPoseToRectsCalculator);

namespace {

constexpr int kWrist = 0;
constexpr int kPinky = 1;
constexpr int kIndex = 2;

constexpr char kImageSizeTag[] = "IMAGE_SIZE";

}  // namespace

::absl::Status HandDetectionsFromPoseToRectsCalculator::Open(
    CalculatorContext* cc) {
  RET_CHECK(cc->Inputs().HasTag(kImageSizeTag))
      << "Image size is required to calculate rotated rect.";
  cc->SetOffset(TimestampDiff(0));
  target_angle_ = M_PI * 0.5f;
  rotate_ = true;
  options_ = cc->Options<DetectionsToRectsCalculatorOptions>();
  output_zero_rect_for_empty_detections_ =
      options_.output_zero_rect_for_empty_detections();

  return ::absl::OkStatus();
}

struct Point {
    float x, y;
};

inline Point GetAbsoluteKeypoint(const LocationData::RelativeKeypoint& keypoint, const std::pair<int, int>& image_size){
    return {keypoint.x() * image_size.first, keypoint.y() * image_size.second};
}

float CalculateDistance(const Point& a, const Point& b) {
    return std::sqrt(std::pow(b.x - a.x, 2) + std::pow(b.y - a.y, 2));
}

::absl::Status
HandDetectionsFromPoseToRectsCalculator ::DetectionToNormalizedRect(
    const Detection& detection, const DetectionSpec& detection_spec,
    NormalizedRect* rect) {
  const auto& location_data = detection.location_data();
  const auto& image_size = detection_spec.image_size;
  RET_CHECK(image_size) << "Image size is required to calculate rotation";

  Point wrist  = GetAbsoluteKeypoint(location_data.relative_keypoints(kWrist), *detection_spec.image_size);
  Point index = GetAbsoluteKeypoint(location_data.relative_keypoints(kIndex), *detection_spec.image_size);
  Point pinky = GetAbsoluteKeypoint(location_data.relative_keypoints(kPinky), *detection_spec.image_size);
  Point middle = {(2.f * index.x + pinky.x) / 3.f, (2.f * index.y + pinky.y) / 3.f};

  float box_size = 2.0 * CalculateDistance(middle, wrist);

  rect->set_x_center(middle.x / detection_spec.image_size->first);
  rect->set_y_center(middle.y / detection_spec.image_size->second);
  rect->set_width(box_size / detection_spec.image_size->first);
  rect->set_height(box_size / detection_spec.image_size->second);

  return ::absl::OkStatus();
}

absl::Status HandDetectionsFromPoseToRectsCalculator::ComputeRotation(
    const Detection& detection, const DetectionSpec& detection_spec,
    float* rotation) {
  const auto& location_data = detection.location_data();
  const auto& image_size = detection_spec.image_size;
  RET_CHECK(image_size) << "Image size is required to calculate rotation";

  Point wrist = GetAbsoluteKeypoint(location_data.relative_keypoints(kWrist), *detection_spec.image_size);
  Point index = GetAbsoluteKeypoint(location_data.relative_keypoints(kIndex), *detection_spec.image_size);
  Point pinky = GetAbsoluteKeypoint(location_data.relative_keypoints(kPinky), *detection_spec.image_size);
  Point middle = {(2.f * index.x + pinky.x) / 3.f, (2.f * index.y + pinky.y) / 3.f};

  *rotation = NormalizeRadians(target_angle_ - std::atan2(-(middle.y - wrist.y), middle.x - wrist.x));

  return ::absl::OkStatus();
}

}  // namespace mediapipe
