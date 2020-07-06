// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "post_process.h"  // NOLINT
#include <algorithm>
#include <utility>

float GetContourArea(const vector_2d_fp& box, float unclip_ratio) {
  int pts_num = 4;
  float area = 0.0f;
  float dist = 0.0f;
  for (int i = 0; i < pts_num; i++) {
    area += box[i][0] * box[(i + 1) % pts_num][1];
    area -= box[i][1] * box[(i + 1) % pts_num][0];
    dist += sqrtf((box[i][0] - box[(i + 1) % pts_num][0]) *
                      (box[i][0] - box[(i + 1) % pts_num][0]) +
                  (box[i][1] - box[(i + 1) % pts_num][1]) *
                      (box[i][1] - box[(i + 1) % pts_num][1]));
  }
  area = fabs(area / 2.0f);
  return area * unclip_ratio / dist;
}

cv::RotatedRect Unclip(const vector_2d_fp& box) {
  float unclip_ratio = 2.0;
  float distance = GetContourArea(box, unclip_ratio);

  ClipperLib::ClipperOffset offset;
  ClipperLib::Path p;
  p << ClipperLib::IntPoint(static_cast<int>(box[0][0]),
                            static_cast<int>(box[0][1]))
    << ClipperLib::IntPoint(static_cast<int>(box[1][0]),
                            static_cast<int>(box[1][1]))
    << ClipperLib::IntPoint(static_cast<int>(box[2][0]),
                            static_cast<int>(box[2][1]))
    << ClipperLib::IntPoint(static_cast<int>(box[3][0]),
                            static_cast<int>(box[3][1]));
  offset.AddPath(p, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

  ClipperLib::Paths soln;
  offset.Execute(soln, distance);
  std::vector<cv::Point2f> points;
  for (int j = 0; j < soln.size(); j++) {
    for (int i = 0; i < soln[soln.size() - 1].size(); i++) {
      points.emplace_back(soln[j][i].X, soln[j][i].Y);
    }
  }
  cv::RotatedRect res = cv::minAreaRect(points);
  return res;
}

vector_2d_fp Mat2Vec(const cv::Mat& mat) {
  auto** array = new float*[mat.rows];
  vector_2d_fp out(mat.rows, std::vector<float>(mat.cols));
  for (int i = 0; i < mat.rows; ++i) {
    for (int j = 0; j < mat.cols; ++j) {
      out[i][j] = mat.at<float>(i, j);
    }
  }
  return out;
}

void QuickSort(vector_2d_fp* input_ptr, int left, int right) {
  vector_2d_fp& input_data = *input_ptr;
  if (left < right) {
    int i = left, j = right;
    float x = input_data[left][0];
    std::vector<float> tmp = input_data[left];
    while (i < j) {
      while (i < j && input_data[j][0] >= x) {
        j--;
      }
      if (i < j) {
        std::swap(input_data[i++], input_data[j]);
      }
      while (i < j && input_data[i][0] < x) {
        i++;
      }
      if (i < j) {
        std::swap(input_data[j--], input_data[i]);
      }
    }
    input_data[i] = tmp;
    QuickSort(input_ptr, left, i - 1);
    QuickSort(input_ptr, i + 1, right);
  }
}

void QuickSort(std::vector<std::vector<int>>* box_ptr, int l, int r, int axis) {
  std::vector<std::vector<int>>& box = *box_ptr;
  if (l < r) {
    int i = l, j = r;
    int x = box[l][axis];
    std::vector<int> xp(box[l]);
    while (i < j) {
      while (i < j && box[j][axis] >= x) {
        j--;
      }
      if (i < j) {
        std::swap(box[i++], box[j]);
      }
      while (i < j && box[i][axis] < x) {
        i++;
      }
      if (i < j) {
        std::swap(box[j--], box[i]);
      }
    }
    box[i] = xp;
    QuickSort(box_ptr, l, i - 1, axis);
    QuickSort(box_ptr, i + 1, r, axis);
  }
}

std::vector<std::vector<int>> OrderPointsClockwise(
    std::vector<std::vector<int>> box) {
  QuickSort(&box, 0, static_cast<int>(box.size() - 1), 0);
  std::vector<std::vector<int>> leftmost = {box[0], box[1]};
  std::vector<std::vector<int>> rightmost = {box[2], box[3]};

  if (leftmost[0][1] > leftmost[1][1]) {
    std::swap(leftmost[0], leftmost[1]);
  }
  if (rightmost[0][1] > rightmost[1][1]) {
    std::swap(rightmost[0], rightmost[1]);
  }

  std::vector<std::vector<int>> res = {
      leftmost[0], rightmost[0], rightmost[1], leftmost[1]};
  return res;
}

vector_2d_fp GetMiniBoxes(const cv::RotatedRect& box) {
  cv::Mat points;
  cv::boxPoints(box, points);
  vector_2d_fp points_vct = Mat2Vec(points);
  QuickSort(&points_vct, 0, 3);

  vector_2d_fp res = points_vct;
  if (points_vct[3][1] <= points_vct[2][1]) {
    res[1] = points_vct[3];
    res[2] = points_vct[2];
  } else {
    res[1] = points_vct[2];
    res[2] = points_vct[3];
  }
  if (points_vct[1][1] <= points_vct[0][1]) {
    res[0] = points_vct[1];
    res[3] = points_vct[0];
  } else {
    res[0] = points_vct[0];
    res[3] = points_vct[1];
  }
  return res;
}

float BoxScoreFast(const vector_2d_fp& box, const cv::Mat& pred) {
  int width = pred.cols;
  int height = pred.rows;

  float box_x[4] = {box[0][0], box[1][0], box[2][0], box[3][0]};
  float box_y[4] = {box[0][1], box[1][1], box[2][1], box[3][1]};

  int xmin = Clamp(
      static_cast<int>(std::floorf(*(std::min_element(box_x, box_x + 4)))),
      0,
      width - 1);
  int xmax =
      Clamp(static_cast<int>(std::ceilf(*(std::max_element(box_x, box_x + 4)))),
            0,
            width - 1);
  int ymin = Clamp(
      static_cast<int>(std::floorf(*(std::min_element(box_y, box_y + 4)))),
      0,
      height - 1);
  int ymax =
      Clamp(static_cast<int>(std::ceilf(*(std::max_element(box_y, box_y + 4)))),
            0,
            height - 1);

  cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

  cv::Point root_point[4];
  root_point[0] = cv::Point(static_cast<int>(box[0][0]) - xmin,
                            static_cast<int>(box[0][1]) - ymin);
  root_point[1] = cv::Point(static_cast<int>(box[1][0]) - xmin,
                            static_cast<int>(box[1][1]) - ymin);
  root_point[2] = cv::Point(static_cast<int>(box[2][0]) - xmin,
                            static_cast<int>(box[2][1]) - ymin);
  root_point[3] = cv::Point(static_cast<int>(box[3][0]) - xmin,
                            static_cast<int>(box[3][1]) - ymin);
  const cv::Point* ppt[1] = {root_point};
  int npt[] = {4};
  cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

  cv::Rect cropped_rect =
      cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
  cv::Mat cropped_img;
  pred(cropped_rect).copyTo(cropped_img);
  float score = cv::mean(cropped_img, mask)[0];
  return score;
}

vector_3d_int BoxesFromBitmap(const cv::Mat& pred, const cv::Mat& bitmap) {
  const int min_size = 3;
  const int max_candidates = 1000;
  const float box_thresh = 0.5;
  float src_width = static_cast<float>(bitmap.cols);
  float src_height = static_cast<float>(bitmap.rows);
  float dest_width = static_cast<float>(pred.cols);
  float dest_height = static_cast<float>(pred.rows);

  vector_3d_int boxes;
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(
      bitmap, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
  int num_contours =
      contours.size() >= max_candidates ? max_candidates : contours.size();

  for (int i = 0; i < num_contours; i++) {
    cv::RotatedRect box = cv::minAreaRect(contours[i]);
    if (std::min(box.size.height, box.size.width) < min_size) {
      continue;
    }
    vector_2d_fp min_box = GetMiniBoxes(box);
    float score = BoxScoreFast(min_box, pred);
    if (score < box_thresh) {
      continue;
    }

    cv::RotatedRect clip_box = Unclip(min_box);
    if (std::min(clip_box.size.height, clip_box.size.width) < min_size + 2) {
      continue;
    }
    auto clip_min_box = GetMiniBoxes(clip_box);

    std::vector<std::vector<int>> clip_points;
    for (int num_pt = 0; num_pt < 4; num_pt++) {
      float x = Clamp(roundf(clip_min_box[num_pt][0] / src_width * dest_width),
                      0.f,
                      dest_width);
      float y =
          Clamp(roundf(clip_min_box[num_pt][1] / src_height * dest_height),
                0.f,
                dest_height);
      clip_points.push_back({static_cast<int>(x), static_cast<int>(y)});
    }
    boxes.push_back(clip_points);
  }
  return boxes;
}

vector_3d_int FilterTagDetRes(vector_3d_int boxes,
                              float ratio_h,
                              float ratio_w,
                              int img_width,
                              int img_height) {
  vector_3d_int root_points;
  for (size_t n = 0; n < boxes.size(); n++) {
    boxes[n] = OrderPointsClockwise(boxes[n]);
    for (size_t m = 0; m < boxes[0].size(); m++) {
      boxes[n][m][0] = static_cast<int>(boxes[n][m][0] / ratio_w);
      boxes[n][m][0] = std::min(std::max(boxes[n][m][0], 0), img_width - 1);
      boxes[n][m][1] = static_cast<int>(boxes[n][m][1] / ratio_h);
      boxes[n][m][1] = std::min(std::max(boxes[n][m][1], 0), img_height - 1);
    }
  }

  const int rect_wh_threshold = 10;
  for (size_t n = 0; n < boxes.size(); n++) {
    int rect_width =
        static_cast<int>(sqrt(pow(boxes[n][0][0] - boxes[n][1][0], 2) +
                              pow(boxes[n][0][1] - boxes[n][1][1], 2)));
    int rect_height =
        static_cast<int>(sqrt(pow(boxes[n][0][0] - boxes[n][3][0], 2) +
                              pow(boxes[n][0][1] - boxes[n][3][1], 2)));
    if (rect_width <= rect_wh_threshold || rect_height <= rect_wh_threshold) {
      continue;
    }
    root_points.push_back(boxes[n]);
  }
  return root_points;
}
