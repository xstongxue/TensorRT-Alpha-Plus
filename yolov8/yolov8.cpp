#include"yolov8.h"
#include"decode_yolov8.h"

// 在构造函数中初始化颜色
YOLOV8::YOLOV8(const utils::InitParameter& param) : yolo::YOLO(param) {
    // 初始化颜色
    colors.resize(80); // 假设有80个类别
    for (int i = 0; i < 80; ++i) {
        colors[i] = cv::Scalar(rand() % 256, rand() % 256, rand() % 256); // 随机颜色
    }
}
YOLOV8::~YOLOV8()
{
    CHECK(cudaFree(m_output_src_transpose_device));
}

bool YOLOV8::init(const std::vector<unsigned char>& trtFile)
{
    if (trtFile.empty())
    {
        return false;
    }
    // 创建runtime
    this->m_runtime.reset(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if (this->m_runtime == nullptr)
    {
        return false;
    }
    this->m_engine.reset(this->m_runtime->deserializeCudaEngine(trtFile.data(), trtFile.size()));
    // this->m_engine = (nvinfer1::ICudaEngine*)(this->m_runtime->deserializeCudaEngine(trtFile.data(), trtFile.size()));

    if (this->m_engine == nullptr)
    {
        return false;
    }
    this->m_context = std::unique_ptr<nvinfer1::IExecutionContext>(this->m_engine->createExecutionContext());
    if (this->m_context == nullptr)
    {
        return false;
    }
    if (m_param.dynamic_batch)
    {
        this->m_context->setBindingDimensions(0, nvinfer1::Dims4(m_param.batch_size, 3, m_param.dst_h, m_param.dst_w));
    }
    m_output_dims = this->m_context->getBindingDimensions(1);
    m_total_objects = m_output_dims.d[2];
    assert(m_param.batch_size <= m_output_dims.d[0]);
    m_output_area = 1;
    for (int i = 1; i < m_output_dims.nbDims; i++)
    {
        if (m_output_dims.d[i] != 0)
        {
            m_output_area *= m_output_dims.d[i];
        }
    }
    CHECK(cudaMalloc(&m_output_src_device, m_param.batch_size * m_output_area * sizeof(float)));
    CHECK(cudaMalloc(&m_output_src_transpose_device, m_param.batch_size * m_output_area * sizeof(float)));
    float a = float(m_param.dst_h) / m_param.src_h;
    float b = float(m_param.dst_w) / m_param.src_w;
    float scale = a < b ? a : b;
    cv::Mat src2dst = (cv::Mat_<float>(2, 3) << scale, 0.f, (-scale * m_param.src_w + m_param.dst_w + scale - 1) * 0.5,
        0.f, scale, (-scale * m_param.src_h + m_param.dst_h + scale - 1) * 0.5);
    cv::Mat dst2src = cv::Mat::zeros(2, 3, CV_32FC1);
    cv::invertAffineTransform(src2dst, dst2src);

    m_dst2src.v0 = dst2src.ptr<float>(0)[0];
    m_dst2src.v1 = dst2src.ptr<float>(0)[1];
    m_dst2src.v2 = dst2src.ptr<float>(0)[2];
    m_dst2src.v3 = dst2src.ptr<float>(1)[0];
    m_dst2src.v4 = dst2src.ptr<float>(1)[1];
    m_dst2src.v5 = dst2src.ptr<float>(1)[2];

    return true;
}

void YOLOV8::preprocess(const std::vector<cv::Mat>& imgsBatch)
{
    resizeDevice(m_param.batch_size, m_input_src_device, m_param.src_w, m_param.src_h,
        m_input_resize_device, m_param.dst_w, m_param.dst_h, 114, m_dst2src);
    bgr2rgbDevice(m_param.batch_size, m_input_resize_device, m_param.dst_w, m_param.dst_h,
        m_input_rgb_device, m_param.dst_w, m_param.dst_h);
    normDevice(m_param.batch_size, m_input_rgb_device, m_param.dst_w, m_param.dst_h,
        m_input_norm_device, m_param.dst_w, m_param.dst_h, m_param);
    hwc2chwDevice(m_param.batch_size, m_input_norm_device, m_param.dst_w, m_param.dst_h,
        m_input_hwc_device, m_param.dst_w, m_param.dst_h);
}


void YOLOV8::postprocess(const std::vector<cv::Mat>& imgsBatch) {
    yolov8::transposeDevice(m_param, m_output_src_device, m_total_objects, 4 + m_param.num_class, m_total_objects * (4 + m_param.num_class), m_output_src_transpose_device, 4 + m_param.num_class, m_total_objects);
    yolov8::decodeDevice(m_param, m_output_src_transpose_device, 4 + m_param.num_class, m_total_objects, m_output_area, m_output_objects_device, m_output_objects_width, m_param.topK);
    nmsDeviceV2(m_param, m_output_objects_device, m_output_objects_width, m_param.topK, m_param.topK * m_output_objects_width + 1, m_output_idx_device, m_output_conf_device);
    
    CHECK(cudaMemcpy(m_output_objects_host, m_output_objects_device, m_param.batch_size * sizeof(float) * (1 + 7 * m_param.topK), cudaMemcpyDeviceToHost));
    
    for (size_t bi = 0; bi < imgsBatch.size(); bi++) {
        int num_boxes = std::min((int)(m_output_objects_host + bi * (m_param.topK * m_output_objects_width + 1))[0], m_param.topK);
        cv::Mat output_img = imgsBatch[bi].clone();
        
        for (size_t i = 0; i < num_boxes; i++) {
            float* ptr = m_output_objects_host + bi * (m_param.topK * m_output_objects_width + 1) + m_output_objects_width * i + 1;
            int keep_flag = ptr[6];
            if (keep_flag) {
                int class_id = static_cast<int>(ptr[5]);
                float x_lt = m_dst2src.v0 * ptr[0] + m_dst2src.v1 * ptr[1] + m_dst2src.v2;
                float y_lt = m_dst2src.v3 * ptr[0] + m_dst2src.v4 * ptr[1] + m_dst2src.v5;
                float x_rb = m_dst2src.v0 * ptr[2] + m_dst2src.v1 * ptr[3] + m_dst2src.v2;
                float y_rb = m_dst2src.v3 * ptr[2] + m_dst2src.v4 * ptr[3] + m_dst2src.v5;

                cv::rectangle(output_img, cv::Point(x_lt, y_lt), cv::Point(x_rb, y_rb), colors[class_id], 2);

                // 构建类别和置信度文本
                std::string label = "Class: " + std::to_string(class_id) + " Conf: " + std::to_string(ptr[4]);
                int baseLine;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                
                // 确保文本框的位置合理
                cv::rectangle(output_img, cv::Point(x_lt, y_lt), cv::Point(x_lt + labelSize.width, y_lt - labelSize.height - baseLine), colors[class_id], cv::FILLED);
                cv::putText(output_img, label, cv::Point(x_lt, y_lt), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            }
        }

        std::string filename = "output_" + std::to_string(bi) + ".jpg"; // 自定义文件名格式
        cv::imwrite(filename, output_img);
    }
}


std::vector<std::tuple<int,float,float,float,float>> YOLOV8::postprocessGetbox(const std::vector<cv::Mat>& imgsBatch) {
    yolov8::transposeDevice(m_param, m_output_src_device, m_total_objects, 4 + m_param.num_class, m_total_objects * (4 + m_param.num_class), m_output_src_transpose_device, 4 + m_param.num_class, m_total_objects);
    yolov8::decodeDevice(m_param, m_output_src_transpose_device, 4 + m_param.num_class, m_total_objects, m_output_area, m_output_objects_device, m_output_objects_width, m_param.topK);
    nmsDeviceV2(m_param, m_output_objects_device, m_output_objects_width, m_param.topK, m_param.topK * m_output_objects_width + 1, m_output_idx_device, m_output_conf_device);
    
    CHECK(cudaMemcpy(m_output_objects_host, m_output_objects_device, m_param.batch_size * sizeof(float) * (1 + 7 * m_param.topK), cudaMemcpyDeviceToHost));
   
    // 存储所有目标的边框和类别
    std::vector<std::tuple<int, float, float, float, float>> all_boxes;
    for (size_t bi = 0; bi < imgsBatch.size(); bi++) {
        int num_boxes = std::min((int)(m_output_objects_host + bi * (m_param.topK * m_output_objects_width + 1))[0], m_param.topK);
        cv::Mat output_img = imgsBatch[bi].clone();

        for (size_t i = 0; i < num_boxes; i++) {
            float* ptr = m_output_objects_host + bi * (m_param.topK * m_output_objects_width + 1) + m_output_objects_width * i + 1;
            int keep_flag = ptr[6];
            if (keep_flag) {
                int class_id = static_cast<int>(ptr[5]);
                float x_lt = m_dst2src.v0 * ptr[0] + m_dst2src.v1 * ptr[1] + m_dst2src.v2;
                float y_lt = m_dst2src.v3 * ptr[0] + m_dst2src.v4 * ptr[1] + m_dst2src.v5;
                float x_rb = m_dst2src.v0 * ptr[2] + m_dst2src.v1 * ptr[3] + m_dst2src.v2;
                float y_rb = m_dst2src.v3 * ptr[2] + m_dst2src.v4 * ptr[3] + m_dst2src.v5;

                // 将边框和类别存储到 all_boxes 中
                all_boxes.push_back(std::make_tuple(class_id, x_lt, y_lt, x_rb, y_rb));
            }
        }
    }
    // 返回包含边框信息和类别的向量
    return all_boxes;
}