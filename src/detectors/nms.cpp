/******************************************************************
 * PyTorch to executable program (torch2exe).
 * Copyright © 2022 Zhiwei Zeng
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy 
 * of this software and associated documentation files (the “Software”), to deal 
 * in the Software without restriction, including without limitation the rights 
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
 * copies of the Software, and to permit persons to whom the Software is 
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all 
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
 * SOFTWARE.
 *
 * This file is part of torch2exe.
 *
 * @file
 * @brief Non-maximum suppression.
 *
 * @author Zhiwei Zeng
 * @email chinaybtseng@gmail.com
 * @version 1.0
 * @date April 11, 2022
 * @license The MIT License (MIT)
 *
 ******************************************************************/

#include <map>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "nms.h"

namespace algorithm {

using id_score_t  = std::pair<int, float>;
using id_score_vt = std::vector<id_score_t>;

static bool cmp(const id_score_t &a, const id_score_t &b)
{
	return a.second > b.second;
}

static inline float iou(const Box &a, const Box &b)
{
	cv::Rect_<float> ra(a.x - a.width * .5f, a.y - a.height * .5f, a.width, a.height);
	cv::Rect_<float> rb(b.x - b.width * .5f, b.y - b.height * .5f, b.width, b.height);
	float inter = (ra & rb).area();
	if (inter == 0) {
		return 0;
	}
	return inter / (ra.area() + rb.area() - inter);
}

void nms(const std::vector<Detection> &dets, std::vector<int> &picked, float score_thresh, float nms_thresh)
{
	std::map<float, id_score_vt> id_scores;
	for (int id = 0; id < static_cast<int>(dets.size()); ++id) {
		if (dets[id].score < score_thresh)
			continue;
		const float &key = dets[id].category;
		auto search = id_scores.find(key);
		if (search == id_scores.end())
			id_scores.emplace(key, id_score_vt());
		id_scores[key].emplace_back(std::make_pair(id, dets[id].score));
	}
	
	for (auto it = id_scores.begin(); it != id_scores.end(); ++it) {
		auto &id_score = it->second;
		std::sort(id_score.begin(), id_score.end(), cmp);
		for (size_t i = 0; i < id_score.size(); ++i) {
			auto &best = id_score[i];
			picked.emplace_back(best.first);
			auto &best_box = dets[best.first].box;
			for (size_t j = i + 1; j < id_score.size(); ++j) {
				auto &test = id_score[j];
				auto &test_box = dets[test.first].box;
				if (iou(best_box, test_box) > nms_thresh) {
					id_score.erase(id_score.begin() + j);
					--j;
				}
			}
		}
	}
}

} // namespace algorithm