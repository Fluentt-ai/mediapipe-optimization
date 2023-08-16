// Copyright 2020 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <cmath>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

using ::mediapipe::NormalizedRect;

namespace {

// NORM_LANDMARKS is either the full set of landmarks for the hand, or
// a subset of the hand landmarks (indices 0, 1, 2, 3, 5, 6, 9, 10, 13, 14,
// 17 and 18). The latter is the legacy behavior, please just pass in
// the full set of hand landmarks.
//
// TODO: update clients to just pass all the landmarks in.
constexpr char kNormalizedLandmarksTag[] = "NORM_LANDMARKS";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
// Indices within the partial landmarks.
constexpr int kWristJoint = 0;
constexpr int kMiddleFingerPIPJoint = 6;
constexpr int kIndexFingerPIPJoint = 4;
constexpr int kRingFingerPIPJoint = 8;
constexpr int kNumLandmarks = 21;
constexpr float kTargetAngle = M_PI * 0.5f;

inline float NormalizeRadians(float angle) {
  return angle - 2 * M_PI * std::floor((angle - (-M_PI)) / (2 * M_PI));
}



std::pair<float, float> RotatePoint(float x, float y, float angle) {
  return std::make_pair(
      x * std::cos(angle) - y * std::sin(angle),
      x * std::sin(angle) + y * std::cos(angle)
  );
}


float ComputeRotation(const NormalizedLandmarkList& landmarks,
                      const std::pair<int, int>& image_size) {
  const float x0 = landmarks.landmark(kWristJoint).x() * image_size.first;
  const float y0 = landmarks.landmark(kWristJoint).y() * image_size.second;

  float x1 = (landmarks.landmark(kIndexFingerPIPJoint).x() +
              landmarks.landmark(kRingFingerPIPJoint).x()) /
             2.f;
  float y1 = (landmarks.landmark(kIndexFingerPIPJoint).y() +
              landmarks.landmark(kRingFingerPIPJoint).y()) /
             2.f;
  x1 = (x1 + landmarks.landmark(kMiddleFingerPIPJoint).x()) / 2.f *
       image_size.first;
  y1 = (y1 + landmarks.landmark(kMiddleFingerPIPJoint).y()) / 2.f *
       image_size.second;

  const float rotation =
      NormalizeRadians(kTargetAngle - std::atan2(-(y1 - y0), x1 - x0));
  return rotation;
}

void ComputeBoundaries(
    const NormalizedLandmarkList& landmarks, float& min_x, float& max_x,
    float& min_y, float& max_y, const std::pair<int, int>& image_size = {1, 1}, 
    float rotation = 0.0f, float center_x_offset = 0.0f, float center_y_offset = 0.0f) {
  for (int i = 0; i < landmarks.landmark_size(); ++i) {
    float x = landmarks.landmark(i).x() * image_size.first - center_x_offset;
    float y = landmarks.landmark(i).y() * image_size.second - center_y_offset;

    if (rotation != 0.0f) {
      std::tie(x, y) = RotatePoint(x, y, rotation);
    }

    max_x = std::max(max_x, x);
    max_y = std::max(max_y, y);
    min_x = std::min(min_x, x);
    min_y = std::min(min_y, y);
  }
}

absl::Status NormalizedLandmarkListToRect(
    const NormalizedLandmarkList& landmarks,
    const std::pair<int, int>& image_size, NormalizedRect* rect) {
  
  const float rotation = ComputeRotation(landmarks, image_size);
  const float reverse_angle = NormalizeRadians(-rotation);
  const float cos_reverse = std::cos(reverse_angle);
  const float sin_reverse = std::sin(reverse_angle);

  float axis_aligned_min_x = std::numeric_limits<float>::max();
  float axis_aligned_min_y = std::numeric_limits<float>::max();
  float axis_aligned_max_x = std::numeric_limits<float>::min();
  float axis_aligned_max_y = std::numeric_limits<float>::min();

  float rotated_min_x = std::numeric_limits<float>::max();
  float rotated_min_y = std::numeric_limits<float>::max();
  float rotated_max_x = std::numeric_limits<float>::min();
  float rotated_max_y = std::numeric_limits<float>::min();

  for (int i = 0; i < landmarks.landmark_size(); ++i) {
    float x = landmarks.landmark(i).x();
    float y = landmarks.landmark(i).y();

    axis_aligned_max_x = std::max(axis_aligned_max_x, x);
    axis_aligned_max_y = std::max(axis_aligned_max_y, y);
    axis_aligned_min_x = std::min(axis_aligned_min_x, x);
    axis_aligned_min_y = std::min(axis_aligned_min_y, y);

    // Calculate the rotated coordinates.
    float dx = (x - (axis_aligned_max_x + axis_aligned_min_x) / 2.0f) * image_size.first;
    float dy = (y - (axis_aligned_max_y + axis_aligned_min_y) / 2.0f) * image_size.second;

    float rotated_x = dx * cos_reverse - dy * sin_reverse;
    float rotated_y = dx * sin_reverse + dy * cos_reverse;

    rotated_max_x = std::max(rotated_max_x, rotated_x);
    rotated_max_y = std::max(rotated_max_y, rotated_y);
    rotated_min_x = std::min(rotated_min_x, rotated_x);
    rotated_min_y = std::min(rotated_min_y, rotated_y);
  }

  float rotated_center_x = (rotated_max_x + rotated_min_x) / 2.0f;
  float rotated_center_y = (rotated_max_y + rotated_min_y) / 2.0f;

  const float cos_rot = std::cos(rotation);
  const float sin_rot = std::sin(rotation);
  
  const float center_x = rotated_center_x * cos_rot -
                         rotated_center_y * sin_rot +
                         image_size.first * (axis_aligned_max_x + axis_aligned_min_x) / 2.0f;
  const float center_y = rotated_center_x * sin_rot +
                         rotated_center_y * cos_rot +
                         image_size.second * (axis_aligned_max_y + axis_aligned_min_y) / 2.0f;

  const float width = (rotated_max_x - rotated_min_x) / image_size.first;
  const float height = (rotated_max_y - rotated_min_y) / image_size.second;

  rect->set_x_center(center_x / image_size.first);
  rect->set_y_center(center_y / image_size.second);
  rect->set_width(width);
  rect->set_height(height);
  rect->set_rotation(rotation);

  return absl::OkStatus();
}


}  // namespace

// A calculator that converts subset of hand landmarks to a bounding box
// NormalizedRect. The rotation angle of the bounding box is computed based on
// 1) the wrist joint and 2) the average of PIP joints of index finger, middle
// finger and ring finger. After rotation, the vector from the wrist to the mean
// of PIP joints is expected to be vertical with wrist at the bottom and the
// mean of PIP joints at the top.
class HandLandmarksToRectCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag(kNormalizedLandmarksTag).Set<NormalizedLandmarkList>();
    cc->Inputs().Tag(kImageSizeTag).Set<std::pair<int, int>>();
    cc->Outputs().Tag(kNormRectTag).Set<NormalizedRect>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    if (cc->Inputs().Tag(kNormalizedLandmarksTag).IsEmpty()) {
      return absl::OkStatus();
    }
    RET_CHECK(!cc->Inputs().Tag(kImageSizeTag).IsEmpty());

    std::pair<int, int> image_size =
        cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();
    const auto landmarks = GetPartialLandmarks(cc);
    auto output_rect = absl::make_unique<NormalizedRect>();
    MP_RETURN_IF_ERROR(
        NormalizedLandmarkListToRect(landmarks, image_size, output_rect.get()));
    cc->Outputs()
        .Tag(kNormRectTag)
        .Add(output_rect.release(), cc->InputTimestamp());

    return absl::OkStatus();
  }

 private:
  NormalizedLandmarkList GetPartialLandmarks(CalculatorContext* cc) {
    const auto& landmarks =
        cc->Inputs().Tag(kNormalizedLandmarksTag).Get<NormalizedLandmarkList>();
    if (landmarks.landmark_size() == kNumLandmarks) {
      static constexpr int kPartialLandmarkIndices[]{0, 1,  2,  3,  5,  6,
                                                     9, 10, 13, 14, 17, 18};
      NormalizedLandmarkList partial_landmarks;
      for (int i : kPartialLandmarkIndices) {
        *partial_landmarks.add_landmark() = landmarks.landmark(i);
      }
      return partial_landmarks;
    } else {
      // Assume the calculator is receiving the partial landmarks directly.
      // This is the legacy behavior.
      return landmarks;
    }
  }
};
REGISTER_CALCULATOR(HandLandmarksToRectCalculator);

}  // namespace mediapipe
