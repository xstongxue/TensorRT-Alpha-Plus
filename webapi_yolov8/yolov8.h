#pragma once
#include"../utils/yolo.h"
#include"../utils/utils.h"
class YOLOV8 : public yolo::YOLO
{
public:
	YOLOV8(const utils::InitParameter& param);
	~YOLOV8();
	virtual bool init(const std::vector<unsigned char>& trtFile);
	virtual void preprocess(const std::vector<cv::Mat>& imgsBatch);
	virtual void postprocess(const std::vector<cv::Mat>& imgsBatch);
	virtual std::vector<std::tuple<std::string, float, float, float, float, float>> postprocessGetbox(const std::vector<cv::Mat>& imgsBatch);
	virtual std::vector<std::tuple<std::string, float, float, float, float>> postprocessWriteIMG(const std::vector<cv::Mat>& imgsBatch);
	virtual void saveImage(const std::vector<cv::Mat>& imgsBatch, const std::vector<std::tuple<std::string, float, float, float, float, float>>& all_boxes);

private:
	float* m_output_src_transpose_device;
	std::vector<cv::Scalar> colors; // 用于存储每个类别的颜色
};