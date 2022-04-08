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

void nmsv2(const std::vector<Detection> &dets, std::vector<int> &picked, float score_thresh, float nms_thresh)
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