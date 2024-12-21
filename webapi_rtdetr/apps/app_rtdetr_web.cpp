
#include <fstream>
#include <opencv2/opencv.hpp>
#include "rtdetr_web.hpp"
#include <chrono>
#include "crow.h"
#include <string_view>
using namespace std;


inline std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v){
    const int h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f*s);
    const float t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (h_i) {
        case 0:r = v; g = t; b = p;break;
        case 1:r = q; g = v; b = p;break;
        case 2:r = p; g = v; b = t;break;
        case 3:r = p; g = q; b = v;break;
        case 4:r = t; g = p; b = v;break;
        case 5:r = v; g = p; b = q;break;
        default:r = 1; g = 1; b = 1;break;}
    return make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
}

inline std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id){
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}

inline string get_file_name(const string& path, bool include_suffix){
    if (path.empty()) return "";
    int p = path.rfind('/');
    int e = path.rfind('\\');
    p = std::max(p, e);
    p += 1;
    //include suffix
    if (include_suffix)
        return path.substr(p);
    int u = path.rfind('.');
    if (u == -1)
        return path.substr(p);

    if (u <= p) u = path.size();
    return path.substr(p, u - p);
}


void performance(const std::string& img_root, const string& engine_file, int gpuid){
    auto infer = RTDETRWEB::create_infer(engine_file, gpuid, 0.5);
    if(infer == nullptr){
        printf("infer is nullptr.\n");
        return;
    }

    int batch = 8;
    std::vector<cv::Mat> images{cv::imread(img_root+"data/imgs/bus.jpg"), cv::imread(img_root+"data/imgs/girl.jpg"),
                                cv::imread(img_root+"data/imgs/group.jpg"), cv::imread(img_root+"data/imgs/yq.jpg")};
    for (int i = images.size(); i < batch; ++i)
        images.push_back(images[i % 4]);

    // warmup
    vector<shared_future<RTDETRWEB::BoxArray>> boxes_array;
    for(int i = 0; i < 10; ++i)
        boxes_array = infer->commits_getBoxes(images);
    boxes_array.back().get();
    boxes_array.clear();

    // 测试 100 轮
    const int ntest = 100;
    auto start = std::chrono::steady_clock::now();
    for(int i  = 0; i < ntest; ++i)
        boxes_array = infer->commits_getBoxes(images);
    // 等待全部推理结束
    boxes_array.back().get();

    std::chrono::duration<double> during = std::chrono::steady_clock::now() - start;
    double all_time = 1000.0 * during.count();
    float avg_time = all_time / ntest / images.size();
    printf("Average time: %.2f ms, FPS: %.2f\n", engine_file.c_str(), avg_time, 1000 / avg_time);
}


void batch_inference(const std::string& img_root, const string& engine_file, int gpuid){
    auto infer = RTDETRWEB::create_infer(engine_file, gpuid, 0.5);
    if(infer == nullptr){
        printf("infer is nullptr.\n");
        return;
    }

    vector<cv::String> files_;
    files_.reserve(100);
    cv::glob(img_root +"data/*.jpg", files_, true);
    vector<string> files(files_.begin(), files_.end());

    vector<cv::Mat> images;
    for(const auto& file : files){
        auto image = cv::imread(file);
        images.emplace_back(image);
    }

    vector<shared_future<RTDETRWEB::BoxArray>> boxes_array;
    boxes_array = infer->commits_getBoxes(images);

    // 等待全部推理结束
    boxes_array.back().get();

    string root_res = img_root+"build";
    for(int i = 0; i < boxes_array.size(); ++i){
        cv::Mat image = images[i];
        auto boxes = boxes_array[i].get();
        for(auto & ibox : boxes){
            cv::Scalar color;
            std::tie(color[0], color[1], color[2]) = random_color(ibox.label);
            cv::rectangle(image, cv::Point(ibox.left, ibox.top), cv::Point(ibox.right, ibox.bottom), color, 2);

            auto name = cocolabels[ibox.label];
            // auto caption = cv::format("%s %.2f", name.c_str(), ibox.confidence);  // 使用string
            auto caption = cv::format("%s %.2f", name.data(), ibox.confidence);  // 使用string_view
            int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(image, cv::Point(ibox.left-2, ibox.top-32), cv::Point(ibox.left + text_width, ibox.top), color, -1);
            cv::putText(image, caption, cv::Point(ibox.left, ibox.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
        }
        string file_name = get_file_name(files[i], false);
        string save_path = cv::format("%s/%s.jpg", root_res.c_str(), file_name.c_str());
        cv::imwrite(save_path, image);
        printf("Save to %s, %d object\n", save_path.c_str(), boxes.size());
    }
}


void single_inference(const std::string& img_root, const string& engine_file, int gpuid){
    auto infer = RTDETRWEB::create_infer(engine_file, gpuid, 0.35);
    if(infer == nullptr){
        printf("infer is nullptr.\n");
        return;
    }

    auto image = cv::imread(img_root + "data/imgs/210114162023244626.jpg");
    auto boxes = infer->commit_getBoxes(image).get();

    for(auto& ibox : boxes){
        cv::Scalar color;
        std::tie(color[0], color[1], color[2]) = random_color(ibox.label);
        cv::rectangle(image, cv::Point(ibox.left, ibox.top), cv::Point(ibox.right, ibox.bottom), color, 2);

        // 获取标签文本和置信度
        auto name = cocolabels[ibox.label];
        // auto caption = cv::format("%s %.2f", name.c_str(), ibox.confidence);  // 使用string
        auto caption = cv::format("%s %.2f", name.data(), ibox.confidence);  // 使用string_view

        // 获取文本的尺寸
        int baseline = 0;
        // 0：使用默认的字体。 2：字体缩放系数（1 表示正常大小）。 2：文字的厚度。
        cv::Size textSize = cv::getTextSize(caption, 0, 2, 2, &baseline);

        // 设置矩形框的宽度和高度
        int text_width = textSize.width + 30;  // 文本宽度加上额外的边距（10像素）
        int text_height = textSize.height + 30; // 文本高度加上额外的边距（10像素）
        // 绘制矩形框，使用文本尺寸来调整框的宽度和高度
        cv::rectangle(image, cv::Point(ibox.left, ibox.top - text_height), cv::Point(ibox.left + text_width, ibox.top), color, 3); 
        // 绘制文本
        cv::putText(image, caption, cv::Point(ibox.left + 5, ibox.top - 20),  // 调整文本的位置，使其位于矩形框内
                    0, 2,  // 文字的颜色、字体大小、厚度
                    cv::Scalar::all(0), 2, 16); 

    }
    cv::imwrite(img_root + "build/result.jpg", image);
    cout << img_root + "build/result.jpg" << endl;
}

void inferWriteIMG(const cv::Mat& img, const string& engine_file, int gpuid){
    auto infer = RTDETRWEB::create_infer(engine_file, gpuid, 0.35);
    if(infer == nullptr){
        printf("infer is nullptr.\n");
        return;
    }

    auto boxes = infer->commit_getBoxes(img).get();

    for(auto& ibox : boxes){
        cv::Scalar color;
        std::tie(color[0], color[1], color[2]) = random_color(ibox.label);
        cv::rectangle(img, cv::Point(ibox.left, ibox.top), cv::Point(ibox.right, ibox.bottom), color, 2);

        // 获取标签文本和置信度
        auto name = cocolabels[ibox.label];
        // auto caption = cv::format("%s %.2f", name.c_str(), ibox.confidence);  // 使用string
        auto caption = cv::format("%s %.2f", name.data(), ibox.confidence);  // 使用string_view

        // 获取文本的尺寸
        int baseline = 0;
        // 0：使用默认的字体。 2：字体缩放系数（1 表示正常大小）。 2：文字的厚度。
        cv::Size textSize = cv::getTextSize(caption, 0, 2, 2, &baseline);

        // 设置矩形框的宽度和高度
        int text_width = textSize.width + 30;  // 文本宽度加上额外的边距（10像素）
        int text_height = textSize.height + 30; // 文本高度加上额外的边距（10像素）
        // 绘制矩形框，使用文本尺寸来调整框的宽度和高度
        cv::rectangle(img, cv::Point(ibox.left, ibox.top - text_height), cv::Point(ibox.left + text_width, ibox.top), color, 3); 
        // 绘制文本
        cv::putText(img, caption, cv::Point(ibox.left + 5, ibox.top - 20),  // 调整文本的位置，使其位于矩形框内
                    0, 2,  // 文字的颜色、字体大小、厚度
                    cv::Scalar::all(0), 2, 16); 
    }
    cv::imwrite("result.jpg", img);
    cout << "save_path: build/result.jpg" << endl;
}

RTDETRWEB::BoxArray inferGetBoxes(const cv::Mat& img, const std::string& engine_file, int gpuid) {
    auto infer = RTDETRWEB::create_infer(engine_file, gpuid, 0.35); // 模型，GPU，阈值
    if (infer == nullptr) {
        std::cerr << "infer is nullptr.\n";
        return {};
    }

    // 获取推理结果
    auto boxes = infer->commit_getBoxes(img).get();
    return boxes;
}


// inferGetIMG 函数
std::string inferGetIMGBase64(const cv::Mat& img, const std::string& engine_file, int gpuid) {
    // 创建推理对象
    auto infer = RTDETRWEB::create_infer(engine_file, gpuid, 0.35);
    if (infer == nullptr) {
        throw std::runtime_error("Infer object creation failed.");
    }

    // 执行推理获取检测框
    auto boxes = infer->commit_getBoxes(img).get();

    // 绘制检测框
    for (auto& ibox : boxes) {
        cv::Scalar color;
        std::tie(color[0], color[1], color[2]) = random_color(ibox.label);
        cv::rectangle(img, cv::Point(ibox.left, ibox.top), cv::Point(ibox.right, ibox.bottom), color, 2);

        // 标签文本和置信度
        auto name = cocolabels[ibox.label];
        // auto caption = cv::format("%s %.2f", name.c_str(), ibox.confidence);  // 使用string
        auto caption = cv::format("%s %.2f", name.data(), ibox.confidence);  // 使用string_view

        // 绘制文本框和文本
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(caption, 0, 2, 2, &baseline);
        cv::rectangle(img, cv::Point(ibox.left, ibox.top - textSize.height - 10),
                      cv::Point(ibox.left + textSize.width + 10, ibox.top), color, -1);
        cv::putText(img, caption, cv::Point(ibox.left + 5, ibox.top - 5), 0, 2, cv::Scalar(0, 0, 0), 2);
    }

    // 编码图像为 JPEG
    std::vector<uchar> buf;
    cv::imencode(".jpg", img, buf);

    // 转换 vector 为字符串
    std::string img_data(buf.begin(), buf.end());

    // 转换为 Base64 编码
    return crow::utility::base64encode(reinterpret_cast<const unsigned char*>(img_data.data()), img_data.size());
}


std::string inferGetIMGBinary(const cv::Mat& img, const std::string& engine_file, int gpuid) {
    // 创建推理对象
    auto infer = RTDETRWEB::create_infer(engine_file, gpuid, 0.35);
    if (infer == nullptr) {
        throw std::runtime_error("Infer object creation failed.");
    }

    // 执行推理获取检测框
    auto boxes = infer->commit_getBoxes(img).get();

    // 绘制检测框
    for (auto& ibox : boxes) {
        cv::Scalar color;
        std::tie(color[0], color[1], color[2]) = random_color(ibox.label);
        cv::rectangle(img, cv::Point(ibox.left, ibox.top), cv::Point(ibox.right, ibox.bottom), color, 2);

        // 标签文本和置信度
        auto name = cocolabels[ibox.label];
        // auto caption = cv::format("%s %.2f", name.c_str(), ibox.confidence);  // 使用string
        auto caption = cv::format("%s %.2f", name.data(), ibox.confidence);  // 使用string_view

        // 绘制文本框和文本
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(caption, 0, 2, 2, &baseline);
        cv::rectangle(img, cv::Point(ibox.left, ibox.top - textSize.height - 10),
                      cv::Point(ibox.left + textSize.width + 10, ibox.top), color, -1);
        cv::putText(img, caption, cv::Point(ibox.left + 5, ibox.top - 5), 0, 2, cv::Scalar(0, 0, 0), 2);
    }

    // 编码图像为 JPEG 格式
    std::vector<uchar> buf;
    cv::imencode(".jpg", img, buf);

    // 将图像二进制数据作为 Blob 返回
    return std::string(buf.begin(), buf.end());
}