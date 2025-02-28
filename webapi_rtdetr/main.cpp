#pragma once
#include <opencv2/opencv.hpp>
#include "../webapi_rtdetr/apps/app_rtdetr_web.cpp"
#include "../webapi_rtdetr/apps/rtdetr_web.hpp"
#include "../webapi_rtdetr/apps/threadpool.hpp"
#include "crow.h"
#include "crow/middlewares/cors.h"
#include "../utils/json.hpp" 
using json = nlohmann::json;
#include <future>  // 确保包含 future 头文件

using namespace std;

void performance(std::string img_root, const string& engine_file, int gpuid);
void batch_inference(const string& img_root, const string& engine_file, int gpuid);
void single_inference(const std::string& img_root, const string& engine_file, int gpuid);

void inferWriteIMG(const cv::Mat& img, const string& engine_file, int gpuid);
RTDETRWEB::BoxArray inferGetBoxes(const cv::Mat& img, const string& engine_file, int gpuid);
std::string inferGetIMGBase64(const cv::Mat& img, const string& engine_file, int gpuid);
std::string inferGetIMGBinary(const cv::Mat& img, const string& engine_file, int gpuid);


class InferTask : public Task {
public:
    InferTask(cv::Mat img, const std::string& model_path, int gpuid, json& response_json, std::mutex& json_mutex)
        : img(std::move(img)), model_path(model_path), gpuid(gpuid), response_json(response_json), json_mutex(json_mutex) {}

    void execute() override {
        RTDETRWEB::BoxArray boxes;
        // 获取推理的信息
        boxes = inferGetBoxes(img, model_path, gpuid);
        json boxes_json;
        for (const auto& box : boxes) {
            auto name = cocolabels[box.label];
            json box_json;
            box_json["box"] = { box.left, box.top, box.right, box.bottom };
            box_json["type"] = name;
            box_json["confidence"] = box.confidence;
            boxes_json.push_back(box_json);
        }

        // Construct response
        std::lock_guard<std::mutex> lock(json_mutex);  // Protect access to response_json
        response_json = {
            {"code", 200},
            {"message", "success"},
            {"data", {{"json", boxes_json}}}
        };

        // Release resources
        img.release();  // 显式释放
        boxes.clear();  // 显示释放
    }

private:
    cv::Mat img;
    std::string model_path;
    int gpuid;
    json& response_json;
    std::mutex& json_mutex;  // Mutex for thread-safe access to response_json
};

int main(int argc, char* argv[]){
    try 
	{
        // 模型路径和GPU通过bash传入，不要暴露给浏览器
        std::string model_path = (argc>1) ? argv[1]: "../weights/rtdetr-l.trt";  // 默认值
        int gpuid = (argc>2) ? std::stoi(argv[2]) : 1;
        int portid = (argc>3) ? std::stoi(argv[3]) : 8888;

        std::string is_save = "false";
		std::string is_show = "false";
		
		// crow::SimpleApp app;
		// crow::Crow<> app;
		// crow::App<crow::> app;
		crow::App<crow::CORSHandler> app;
		// Customize CORS
		auto& cors = app.get_middleware<crow::CORSHandler>();

		// clang-format off
		cors
		.global()
			.headers("X-Custom-Header", "Upgrade-Insecure-Requests")
			.methods("POST"_method, "GET"_method)
		.prefix("/cors")
			.origin("192.168.4.9:8889")
		.prefix("/nocors")
			.ignore();

		ThreadPool thread_pool(4);  // 创建线程池，假设我们有4个线程

		// 上传文件处理路由
		CROW_ROUTE(app, "/pic_infer").methods(crow::HTTPMethod::Post, crow::HTTPMethod::Options)([&](const crow::request& req) {

            std::cout << std::endl;
            using json = nlohmann::json;
			json boxes_json = json::array();
			json response_json;

            if(argc<2){
                std::cerr << "Usage: " << argv[0] << " <model_path> <gpu_id> <port>" << std::endl;
                response_json = {
                    {"code", -1},
                    {"message", "Usage: <input_image_path> <output_image_path>"}
                };
                return crow::response(response_json.dump(2));
            }

			// std::cout << "Received request, processing multipart data..." << std::endl;
			crow::multipart::message_view file_message(req);

			// 遍历 multipart 中的各部分
			for (const auto& part : file_message.part_map) {
				const auto& part_name = part.first;
				const auto& part_value = part.second;

				if (part_name == "is_save") 
				{
					is_save = std::string(part_value.body.begin(), part_value.body.end());
				} 
				else if (part_name == "is_show") 
				{
					is_show = std::string(part_value.body.begin(), part_value.body.end());
				} 
				else if ("img" == part_name) 
				{
					std::cout << "Received parameters - model_path: " << model_path
                    << ", gpu: " << gpuid 
                    << ", port: " << portid
					<< ", is_save: " << is_save 
					<< ", is_show: " << is_show << std::endl;
					// std::cout << "Found 'img', processing..." << std::endl;

					// 从请求头中获取文件名
					auto headers_it = part_value.headers.find("Content-Disposition");
					if (headers_it == part_value.headers.end()) {
						std::cout << "Error: Missing 'Content-Disposition' header" << std::endl;
						return crow::response(400);  // 请求头中没有 Content-Disposition
					}
					auto params_it = headers_it->second.params.find("filename");
					if (params_it == headers_it->second.params.end()) {
						std::cout << "Error: Missing filename in 'Content-Disposition' header" << std::endl;
						return crow::response(400);  // 没有文件名
					}
					const std::string filename(params_it->second);
					// std::cout << "File uploaded: " << filename << std::endl;

					// 使用 OpenCV 将上传的文件内容转换为 cv::Mat
					std::vector<uchar> img_data(part_value.body.begin(), part_value.body.end());
					std::vector<uchar> temp_v;	
					cv::Mat img = cv::imdecode(img_data, cv::IMREAD_COLOR);
					if (img.empty()) {
						std::cout << "Error: Failed to decode image" << std::endl;
						return crow::response(400, "Invalid image data");
					}
					// std::cout << "Image decoded successfully..." << std::endl;
                    
                    RTDETRWEB::BoxArray boxes;
					if (is_save=="true")
                    {
                        inferWriteIMG(img, model_path, gpuid);
                        response_json = {
                            {"code", 200},
                            {"message", "success"},
                            {"detect result", "already save to build/result.jpg"}
                        };
                        return crow::response(response_json.dump(4));  // 返回推理结果
                    }
					else if(is_show=="true")
					{
						auto detect_img_binary = inferGetIMGBinary(img, model_path, gpuid);
						if (!detect_img_binary.empty()){
							// Return the encoded result
							crow::response res(200);  // 设置响应状态码为 200 OK
							res.set_header("Content-Type", "image/jpeg");  // 设置响应类型为 image/jpeg
							res.body = detect_img_binary;  // 将二进制数据赋值给响应的 body
							return res;  // 返回响应
						}
						// auto detect_img_base64 = inferGetIMGBase64(img, model_path, gpuid);
						// if(!detect_img_base64.empty()){
						// 	return crow::response(200, detect_img_base64);
						// }
					}
					else
					{
						// 创建并提交任务到线程池（2024.12.20修改版本：加入线程池）
						// 2025.2.17: 创建一个互斥锁，确保线程安全(多个线程之间不会同时访问 response_json )
						std::mutex json_mutex;
						auto task = std::make_shared<InferTask>(img, model_path, gpuid, response_json, json_mutex);
						auto future = thread_pool.submitTask(task);  // 提交任务到线程池，返回future
						// 等待任务完成
						future.get();  // 这里会阻塞，直到任务执行完毕

						img_data.clear();  // 显示释放
						std::swap(img_data, temp_v);  // vector 容器通过clear并没有清空，需要通过swap或者shrink_to_fit
						// 等待任务执行完成，返回响应
						// cout << "response_json: " << response_json.dump(4) << endl;
						return crow::response(response_json.dump(4));
					}
                    // else
                    // {
                    //      // 记录起始时间
                    //     auto start = std::chrono::high_resolution_clock::now();
                    //     // 获取推理的信息
                    //     boxes = inferGetBoxes(img, model_path, gpuid);
                    //     // 记录结束时间
                    //     auto end = std::chrono::high_resolution_clock::now();
                    //     // 计算耗时
                    //     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                    //     std::cout << "IMG Name: " << filename << ", inferGetBoxes took " << duration << " ms.";
                    //     std::cout << ", Detected boxes: " << boxes.size() << std::endl;
                    // }
					// // 构建返回的JSON
					// for (const auto& box : boxes) {
                    //     auto name = cocolabels[box.label];
					// 	json box_json;
					// 	box_json["box"] = {box.left, box.top, box.right, box.bottom};
					// 	box_json["type"] = name;
					// 	box_json["confidence"] = box.confidence;
					// 	boxes_json.push_back(box_json);
					// }

					// response_json = {
					// 	{"code", 200},
					// 	{"message", "success"},
					// 	{"data", {
					// 		{"json", boxes_json},
					// 		// {"img", base64_str}
					// 	}}
					// };
					// // cout << "response_json: " << response_json.dump(4) << endl;
					
					// img.release();  // 显式释放内存
					// boxes.clear();  // 显示释放
					// img_data.clear();  // 显示释放
					// std::swap(img_data, temp_v);  // vector 容器通过clear并没有清空，需要通过swap或者shrink_to_fit
					// return crow::response(response_json.dump(4));  // 返回推理结果
				}
			}

			std::cout << "Error: No file part named 'img' found in request" << std::endl;
			return crow::response(400);  // 未找到文件部分
		});

		app.port(portid).multithreaded().run();  // 启动 HTTP 服务
	} 
	catch (const std::exception& e) 
	{
		std::cerr << "Caught exception: " << e.what() << std::endl;
		return -1;  // 处理异常，确保程序不会意外退出
	}
	return 0;  // 成功结束
}