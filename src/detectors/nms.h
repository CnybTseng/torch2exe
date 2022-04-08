#ifndef NMS_H_
#define NMS_H_

#include "algorithm.h"
#include "operators/plugins/common.h"

namespace algorithm {

void nmsv2(const std::vector<Detection> &dets, std::vector<int> &picked, float score_thresh, float nms_thresh);

} // namespace algorithm

#endif // NMS_H_