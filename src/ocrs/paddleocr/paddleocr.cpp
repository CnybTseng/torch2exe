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
 * @brief PaddleOCR.
 *
 * @author Zhiwei Zeng
 * @email chinaybtseng@gmail.com
 * @version 1.0
 * @date June 27, 2022
 * @license The MIT License (MIT)
 *
 ******************************************************************/

#include <cstring>
#include <limits>

#include <opencv2/core/types.hpp>

#include "utils/configurer.h"
#include "utils/logger.h"
#include "utils/deleter.h"
#include "include/paddleocr.h"

namespace algorithm {
namespace ocr {

std::unique_ptr<Algorithm, Deleter> PPOCR::create(const char *cfg)
{
	LogInfo("PPOCR::create\n");
	return std::unique_ptr<Algorithm, Deleter>(new PPOCR, [](void *p){reinterpret_cast<PPOCR *>(p)->destroy();});
}

std::string PPOCR::get_name(void)
{
	return "PaddleOCR";
}

bool PPOCR::init(const char *cfg)
{
	if (!parse_config(cfg)) {
		return false;
	}
	
	if (det_model_dir != std::string("")) {
		LogInfo("PPOCR detector init ...\n");
		detector = std::shared_ptr<PaddleOCR::DBDetector>(new PaddleOCR::DBDetector(
			det_model_dir, use_gpu, gpu_id, gpu_mem, cpu_math_library_num_threads,
			use_mkldnn, max_side_len, det_db_thresh, det_db_box_thresh, det_db_unclip_ratio,
			det_db_score_mode, use_dilation, use_tensorrt, precision
		));
		LogInfo("PPOCR detector init done\n");
	}
	
	if (cls_model_dir != std::string("")) {
		LogInfo("PPOCR classifier init ...\n");
		classifier = std::shared_ptr<PaddleOCR::Classifier>(new PaddleOCR::Classifier(
			cls_model_dir, use_gpu, gpu_id, gpu_mem, cpu_math_library_num_threads,
			use_mkldnn, cls_thresh, use_tensorrt, precision, cls_batch_num
		));
		LogInfo("PPOCR classifier init done\n");
	}
	
	LogInfo("PPOCR recognizer init ...\n");
	recognizer = std::shared_ptr<PaddleOCR::CRNNRecognizer>(new PaddleOCR::CRNNRecognizer(
		rec_model_dir, use_gpu, gpu_id, gpu_mem, cpu_math_library_num_threads,
		use_mkldnn, label_path, use_tensorrt, precision, rec_batch_num,
		rec_img_h, rec_img_w
	));
	LogInfo("PPOCR recognizer init done\n");
	return true;
}

bool PPOCR::execute(const BlobSP &input, BlobSP &output)
{
	LogError("please call `bool execute(const std::initializer_list<BlobSP> &inputs, BlobSP &output)` instead\n");
	return false;
}

bool PPOCR::execute(const std::initializer_list<BlobSP> &inputs, BlobSP &output)
{
	if (inputs.size() != 2) {
		LogError("incorrect number of input blobs: %zu != 2\n", inputs.size());
		return false;
	}
	
	const ImageSP img = std::static_pointer_cast<Image>(*inputs.begin());
	if (!img) {
		LogError("nullptr to input image\n");
		return false;
	}
	
	const cv::Mat mat(img->height, img->width, CV_8UC3, img->data, img->stride);
	std::vector<cv::Mat> img_list;
	std::vector<PaddleOCR::OCRPredictResult> ocr_result;
	const Object2DBoxFASP dets = std::static_pointer_cast<Object2DBoxFA>(*(inputs.begin() + 1));
	if (dets) {	//!< Use given boxes 
		LogTrace("use given boxes\n");
		for (int i = 0; i < dets->count; ++i) {
			if (std::string(dets->data[i].category) != cate_for_ocr) {
				continue;
			}

			//! Make sure that the rect will not cross the image boundary.
			int bx = dets->data[i].box.x >= 0 ? dets->data[i].box.x : 0;
			int by = dets->data[i].box.y >= 0 ? dets->data[i].box.y : 0;
			const int maxw = mat.cols - 1 - bx;
			const int maxh = mat.cols - 1 - by;
			int bw = dets->data[i].box.width < maxw ? dets->data[i].box.width : maxw;
			int bh = dets->data[i].box.height < maxh ? dets->data[i].box.height : maxh;
			
			cv::Mat roi = mat(cv::Rect(bx, by, bw, bh));
			img_list.push_back(roi);
			
			//! Copy box.
			PaddleOCR::OCRPredictResult res;
			int x = dets->data[i].box.x;
			int y = dets->data[i].box.y;
			int w = dets->data[i].box.width;
			int h = dets->data[i].box.height;
			std::vector<int> xy1 = {x, y};
			std::vector<int> xy2 = {x + w - 1, y};
			std::vector<int> xy3 = {x, y + h - 1};
			std::vector<int> xy4 = {x + w - 1, y + h - 1};
			res.box.push_back(xy1);
			res.box.push_back(xy2);
			res.box.push_back(xy3);
			res.box.push_back(xy4);
			ocr_result.push_back(res);
		}
	} else {
		std::vector<double> time_info_det = {0, 0, 0};
		if (detector) {	//!< Detect boxes
			LogTrace("detect boxes with PPOCR\n");
			det(mat, ocr_result, time_info_det);
			for (int j = 0; j < ocr_result.size(); j++) {
				cv::Mat roi = PaddleOCR::Utility::GetRotateCropImage(mat, ocr_result[j].box);
				img_list.push_back(roi);
			}
		} else {
			LogTrace("use whole image for PPOCR\n");
			img_list.push_back(mat);
			PaddleOCR::OCRPredictResult res;
			int x = 0;
			int y = 0;
			int w = mat.cols;
			int h = mat.rows;
			std::vector<int> xy1 = {x, y};
			std::vector<int> xy2 = {x + w - 1, y};
			std::vector<int> xy3 = {x, y + h - 1};
			std::vector<int> xy4 = {x + w - 1, y + h - 1};
			res.box.push_back(xy1);
			res.box.push_back(xy2);
			res.box.push_back(xy3);
			res.box.push_back(xy4);
			ocr_result.push_back(res);
		}
	}
	
	//! Classification.
	if (classifier) {
		LogTrace("classifier ...\n");
		std::vector<double> time_info_cls = {0, 0, 0};
		cls(img_list, ocr_result, time_info_cls);
		for (int i = 0; i < img_list.size(); i++) {
			if (ocr_result[i].cls_label % 2 == 1 &&
				ocr_result[i].cls_score > classifier->cls_thresh) {
				cv::rotate(img_list[i], img_list[i], 1);
				LogTrace("rotate image\n");
			}
		}
		LogTrace("classifier done\n");
	}
	
	//! Recognition.
	LogTrace("recognizer ...\n");
	std::vector<double> time_info_rec = {0, 0, 0};
	rec(img_list, ocr_result, time_info_rec);
	LogTrace("recognizer done\n");
	
	size_t count = 0;
	for (size_t i = 0; i < ocr_result.size(); ++i) {
		if (ocr_result[i].score > 0) {
			++count;
		}
	}

	if (count < 1) {
		return true;
	}

	size_t size = sizeof(LicensePlateFA) + count * sizeof(LicensePlate);
	output.reset(reinterpret_cast<LicensePlateFA *>(new (std::nothrow) char[size]), ArrayDeleter("detector output"));
	if (!output) {
		LogError("allocate memory failed\n");
		return false;
	}
	
	LicensePlateFASP lps = std::static_pointer_cast<LicensePlateFA>(output);
	lps->type = OCR;
	lps->count = static_cast<uint16_t>(count);
	int k = 0;
	for (int i = 0; i < ocr_result.size(); ++i) {
		if (ocr_result[i].score > 0) {
			strcpy(lps->data[k].text, ocr_result[i].text.c_str());
			lps->data[k].score = ocr_result[i].score;
			int minx = std::numeric_limits<int>::max();
			int maxx = std::numeric_limits<int>::min();
			int miny = std::numeric_limits<int>::max();
			int maxy = std::numeric_limits<int>::min();
			auto &box = ocr_result[i].box;
			for (int j = 0; j < box.size(); ++j) {	//!< Four vertices
				auto &xy = box[j];
				if (xy[0] < minx) {
					minx = xy[0];
				}
				if (xy[0] > maxx) {
					maxx = xy[0];
				}
				if (xy[1] < miny) {
					miny = xy[1];
				}
				if (xy[1] > maxy) {
					maxy = xy[1];
				}
			}
			lps->data[k].box.x = static_cast<uint16_t>(minx);
			lps->data[k].box.y = static_cast<uint16_t>(miny);
			lps->data[k].box.width = static_cast<uint16_t>(maxx - minx + 1);
			lps->data[k].box.height = static_cast<uint16_t>(maxy - miny + 1);
			++k;
		}
	}
	
	return true;
}

void PPOCR::destroy(void)
{
	LogInfo("PPOCR::destroy\n");
}

bool PPOCR::parse_config(const char *cfg)
{
	Configurer cfgr;
	if (!cfgr.init(cfg)) {
		return false;
	}
	
	cate_for_ocr = cfgr.get("cate_for_ocr", std::string("lp"));
	
	//! Common specifications.
	use_gpu = cfgr.get("use_gpu", false);
	gpu_id = cfgr.get("gpu_id", 0);
	gpu_mem = cfgr.get("gpu_mem", int(4000));
	cpu_math_library_num_threads = cfgr.get("cpu_math_library_num_threads", int(10));
	use_mkldnn = cfgr.get("use_mkldnn", false);
	use_tensorrt = cfgr.get("use_tensorrt", false);
	precision = cfgr.get("precision", "fp32");
	
	//! Detector specifications.
	det_model_dir = cfgr.get("det_model_dir", std::string(""));
	if (det_model_dir == "") {
		LogWarn("empty det_model_dir\n");
	}
	max_side_len = cfgr.get("max_side_len", int(960));
	det_db_thresh = cfgr.get("det_db_thresh", double(0.3));
	det_db_box_thresh = cfgr.get("det_db_box_thresh", double(0.6));
	det_db_unclip_ratio = cfgr.get("det_db_unclip_ratio", double(1.5));
	det_db_score_mode = cfgr.get("det_db_score_mode", std::string("slow"));
	use_dilation = cfgr.get("use_dilation", false);
	
	//! Classifier specifications.
	cls_model_dir = cfgr.get("cls_model_dir", std::string(""));
	if (cls_model_dir == "") {
		LogWarn("empty cls_model_dir\n");
	}
	cls_thresh = cfgr.get("cls_thresh", double(0.9));
	cls_batch_num = cfgr.get("cls_batch_num", int(1));
	
	//! Recognizer specifications.
	rec_model_dir = cfgr.get("rec_model_dir", std::string(""));
	if (rec_model_dir == "") {
		LogError("empty rec_model_dir\n");
		return false;
	}
	label_path = cfgr.get("label_path", std::string(""));
	rec_batch_num = cfgr.get("rec_batch_num", int(6));
	rec_img_h = cfgr.get("rec_img_h", int(48));
    rec_img_w = cfgr.get("rec_img_w", int(320));
	
	return true;
}

void PPOCR::det(cv::Mat img, std::vector<PaddleOCR::OCRPredictResult> &ocr_results,
	std::vector<double> &times)
{
	std::vector<std::vector<std::vector<int>>> boxes;
	std::vector<double> det_times;

	detector->Run(img, boxes, det_times);

	for (int i = 0; i < boxes.size(); i++) {
		PaddleOCR::OCRPredictResult res;
		res.box = boxes[i];
		ocr_results.push_back(res);
	}

	times[0] += det_times[0];
	times[1] += det_times[1];
	times[2] += det_times[2];
}

void PPOCR::cls(std::vector<cv::Mat> img_list, std::vector<PaddleOCR::OCRPredictResult> &ocr_results,
	std::vector<double> &times)
{
	std::vector<int> cls_labels(img_list.size(), 0);
	std::vector<float> cls_scores(img_list.size(), 0);
	std::vector<double> cls_times;
	classifier->Run(img_list, cls_labels, cls_scores, cls_times);
	// output cls results
	for (int i = 0; i < cls_labels.size(); i++) {
		ocr_results[i].cls_label = cls_labels[i];
		ocr_results[i].cls_score = cls_scores[i];
	}
	times[0] += cls_times[0];
	times[1] += cls_times[1];
	times[2] += cls_times[2];
}
	
void PPOCR::rec(std::vector<cv::Mat> img_list, std::vector<PaddleOCR::OCRPredictResult> &ocr_results,
	std::vector<double> &times)
{
	std::vector<std::string> rec_texts(img_list.size(), "");
	std::vector<float> rec_text_scores(img_list.size(), 0);
	std::vector<double> rec_times;
	recognizer->Run(img_list, rec_texts, rec_text_scores, rec_times);
	// output rec results
	for (int i = 0; i < rec_texts.size(); i++) {
		ocr_results[i].text = rec_texts[i];
		ocr_results[i].score = rec_text_scores[i];
	}
	times[0] += rec_times[0];
	times[1] += rec_times[1];
	times[2] += rec_times[2];
}
	
}
}