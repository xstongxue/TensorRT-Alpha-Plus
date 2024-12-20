#include <muduo/net/EventLoop.h>
#include <muduo/net/TcpServer.h>
#include <muduo/net/Buffer.h>
#include <muduo/base/Logging.h>  // 包含Logging头文件
#include <opencv2/opencv.hpp>
#include "/data02/xs/code/TensorRT_deploy/tensorrt-alpha-plus/webapi_rtdetr/apps/app_rtdetr_web.cpp"
#include "../utils/json.hpp"
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <exception>  // 用于异常处理

// 手动解析 multipart/form-data 类型有问题，待解决！

using json = nlohmann::json;
using namespace std;
using namespace muduo;
using namespace muduo::net;

// 模型推理相关函数声明
void inferWriteIMG(const cv::Mat& img, const string& engine_file, int gpuid);
RTDETRWEB::BoxArray inferGetBoxes(const cv::Mat& img, const string& engine_file, int gpuid);
std::string inferGetIMGBase64(const cv::Mat& img, const string& engine_file, int gpuid);
std::string inferGetIMGBinary(const cv::Mat& img, const string& engine_file, int gpuid);

extern std::vector<std::string> cocolabels;

// 解析 multipart 数据
bool parseMultipartFormData(const std::string& request, std::string& img_data, std::string& is_save, std::string& is_show)
{
    try {
        size_t boundary_pos = request.find("boundary=");
        if (boundary_pos == std::string::npos) {
            std::cerr << "Error: Missing boundary in multipart data." << std::endl;
            return false;
        }

        std::string boundary = "--" + request.substr(boundary_pos + 9, request.find("\r\n", boundary_pos) - boundary_pos - 9);
        std::cout << "Found boundary: " << boundary << std::endl;

        size_t start_pos = request.find(boundary);
        size_t end_pos = request.find(boundary, start_pos + 1);
        while (start_pos != std::string::npos && end_pos != std::string::npos) {
            std::string part = request.substr(start_pos, end_pos - start_pos);

            std::cout << "Part:\n" << part << "\n" << std::endl;  // 打印出部分内容进行调试

            size_t content_disposition_pos = part.find("Content-Disposition: form-data;");
            if (content_disposition_pos != std::string::npos) {
                std::cout << "Found Content-Disposition header." << std::endl;

                size_t filename_pos = part.find("filename=\"");
                if (filename_pos != std::string::npos) {
                    std::cout << "Found file part." << std::endl;
                    size_t content_start = part.find("\r\n\r\n", content_disposition_pos) + 4;
                    img_data = part.substr(content_start, part.length() - content_start - 2);  // 排除 \r\n\r\n 和 boundary
                    std::cout << "Image data size: " << img_data.size() << std::endl;

                    // 处理参数
                    size_t is_save_pos = part.find("is_save=");
                    if (is_save_pos != std::string::npos) {
                        is_save = part.substr(is_save_pos + 8, part.find("\r\n", is_save_pos) - is_save_pos - 8);
                        std::cout << "is_save: " << is_save << std::endl;
                    }

                    size_t is_show_pos = part.find("is_show=");
                    if (is_show_pos != std::string::npos) {
                        is_show = part.substr(is_show_pos + 8, part.find("\r\n", is_show_pos) - is_show_pos - 8);
                        std::cout << "is_show: " << is_show << std::endl;
                    }
                    return true;
                }
            }

            start_pos = end_pos;
            end_pos = request.find(boundary, start_pos + 1);
        }
        std::cerr << "Error: Failed to find the image part in multipart data." << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing multipart data: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "Unknown error occurred while parsing multipart data." << std::endl;
        return false;
    }
}


void onMessage(const TcpConnectionPtr &conn, Buffer *buf, Timestamp receiveTime)
{
    try {
        string model_path = "../weights/rtdetr-l.trt";  // 默认路径
        int gpuid = 1;  // 默认GPU
        std::string is_save = "false";
        std::string is_show = "false";

        // 解析 HTTP 请求头部
        string request = buf->retrieveAllAsString();
        if (request.find("POST /pic_infer") == std::string::npos) {
            conn->send("HTTP/1.1 404 Not Found\r\n\r\n");
            conn->shutdown();
            return;
        }

        // 解析 multipart 数据
        string img_data;
        if (!parseMultipartFormData(request, img_data, is_save, is_show)) {
            conn->send("HTTP/1.1 400 Bad Request\r\n\r\nInvalid multipart data");
            conn->shutdown();
            return;
        }

        // 使用 OpenCV 读取图片
        std::vector<uchar> img_vector(img_data.begin(), img_data.end());
        cv::Mat img = cv::imdecode(img_vector, cv::IMREAD_COLOR);
        if (img.empty()) {
            conn->send("HTTP/1.1 400 Bad Request\r\n\r\nInvalid image data");
            conn->shutdown();
            return;
        }

        RTDETRWEB::BoxArray boxes;
        json response_json;
        json boxes_json = json::array();

        // 根据 is_save 或 is_show 处理逻辑
        if (is_save == "true") {
            inferWriteIMG(img, model_path, gpuid);
            response_json = {
                {"code", 200},
                {"message", "success"},
                {"detect result", "already saved to build/result.jpg"}
            };
        } else if (is_show == "true") {
            auto detect_img_binary = inferGetIMGBinary(img, model_path, gpuid);
            if (!detect_img_binary.empty()) {
                response_json = {
                    {"code", 200},
                    {"message", "success"},
                    {"image", detect_img_binary}
                };
            }
        } else {
            auto start = std::chrono::high_resolution_clock::now();
            boxes = inferGetBoxes(img, model_path, gpuid);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "Inference took " << duration << " ms." << std::endl;

            for (const auto& box : boxes) {
                auto name = cocolabels[box.label];
                json box_json;
                box_json["box"] = {box.left, box.top, box.right, box.bottom};
                box_json["type"] = name;
                box_json["confidence"] = box.confidence;
                boxes_json.push_back(box_json);
            }
            response_json = {
                {"code", 200},
                {"message", "success"},
                {"data", {{"json", boxes_json}}}
            };
        }

        // 发送响应
        conn->send("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n");
        conn->send(response_json.dump(4));
        conn->shutdown();  // 关闭连接

    } catch (const std::exception& e) {
        std::cerr << "Error in onMessage: " << e.what() << std::endl;
        conn->send("HTTP/1.1 500 Internal Server Error\r\n\r\n");
        conn->shutdown();
    } catch (...) {
        std::cerr << "Unknown error occurred in onMessage." << std::endl;
        conn->send("HTTP/1.1 500 Internal Server Error\r\n\r\n");
        conn->shutdown();
    }
}

// 主函数：创建并启动 TCP 服务器
int main(int argc, char* argv[])
{
    try {
        // 日志初始化
        muduo::Logger::setLogLevel(muduo::Logger::INFO);  // 设置日志级别

        // 设置监听端口和IP地址
        string ip = "0.0.0.0";  // 监听所有 IP 地址
        int port = 8888;  // 监听端口

        // 设置 EventLoop 和 TcpServer
        EventLoop loop;
        InetAddress listenAddr(port);
        TcpServer server(&loop, listenAddr, "RTDETRWebServer");

        // 注册处理函数
        server.setMessageCallback(onMessage);

        // 启动服务器
        server.start();
        loop.loop();  // 进入事件循环
    } catch (const std::exception& e) {
        std::cerr << "Error in main: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown error occurred in main." << std::endl;
        return -1;
    }
    return 0;
}
