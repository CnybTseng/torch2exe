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

#ifndef PADDLEOCR_H_
#define PADDLEOCR_H_

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include <include/ocr_cls.h>
#include <include/ocr_det.h>
#include <include/ocr_rec.h>
#include <include/preprocess_op.h>
#include <include/utility.h>

#include "algorithm.h"
#include "algorithmExt.h"
#include "utils/algorithmFactory.h"

namespace algorithm {
namespace ocr {

class PPOCR : public AlgorithmExt, public RegisteredInFactory<Algorithm, PPOCR>
{
public:
	PPOCR() : detector(nullptr), classifier(nullptr), recognizer(nullptr) {}
	static std::unique_ptr<Algorithm, Deleter> create(const char *cfg);
	static std::string get_name(void);
	virtual ~PPOCR() = default;
	virtual bool init(const char *cfg) override;
	virtual bool execute(const BlobSP &input, BlobSP &output) override;
	virtual bool execute(const std::initializer_list<BlobSP> &inputs, BlobSP &output) override;
	virtual void destroy(void) override;
private:
	virtual bool parse_config(const char *cfg) override;
	
	void det(cv::Mat img, std::vector<PaddleOCR::OCRPredictResult> &ocr_results,
		std::vector<double> &times);
	
	void cls(std::vector<cv::Mat> img_list, std::vector<PaddleOCR::OCRPredictResult> &ocr_results,
		std::vector<double> &times);
	
	void rec(std::vector<cv::Mat> img_list, std::vector<PaddleOCR::OCRPredictResult> &ocr_results,
		std::vector<double> &times);
	
	std::shared_ptr<PaddleOCR::DBDetector> detector;
	std::shared_ptr<PaddleOCR::Classifier> classifier;
	std::shared_ptr<PaddleOCR::CRNNRecognizer> recognizer;
	
	std::string cate_for_ocr;
	
	//! Common specifications.
	bool use_gpu;
	int gpu_id;
	int gpu_mem;
	int cpu_math_library_num_threads;
	bool use_mkldnn;
	bool use_tensorrt;
	std::string precision;
	
	//! Detector specifications.
	std::string det_model_dir;
	int max_side_len;
	double det_db_thresh;
	double det_db_box_thresh;
	double det_db_unclip_ratio;
	std::string det_db_score_mode;
	bool use_dilation;
	
	//! Classifier specifications.
	std::string cls_model_dir;
	double cls_thresh;
	int cls_batch_num;
	
	//! Recognizer specifications.
	std::string rec_model_dir;
	std::string label_path;
	int rec_batch_num;
	int rec_img_h;
    int rec_img_w;
};

} //!< namespace ocr
} //!< namespace algorithm

#endif //!< PADDLEOCR_H_