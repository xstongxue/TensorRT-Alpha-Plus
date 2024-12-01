#include "../utils/yolo.h"
#include "../utils/json.hpp" 
#include "yolov8.h"
#include <vector>
using namespace std;
#include "crow.h"
#include <string_view>
#include <tuple>
using json = nlohmann::json;

void setParameters(utils::InitParameter& initParameters)
{
	initParameters.class_names = utils::dataSets::coco80;
	// initParameters.class_names = utils::dataSets::voc20;
	initParameters.num_class = 80; // for coco
	// initParameters.num_class = 20; // for voc2012
	initParameters.batch_size = 8;
	initParameters.dst_h = 640;
	initParameters.dst_w = 640;
	initParameters.input_output_names = { "images",  "output0" };
	initParameters.conf_thresh = 0.20f;
	initParameters.iou_thresh = 0.45f;
	initParameters.save_path = "";
}

void task(std::unique_ptr<YOLOV8>& yolo, const utils::InitParameter& param, std::vector<cv::Mat>& imgsBatch, const int& delayTime, const int& batchi,
	const bool& isShow, const bool& isSave)
{
	utils::DeviceTimer d_t0; yolo->copy(imgsBatch);	      float t0 = d_t0.getUsedTime();
	utils::DeviceTimer d_t1; yolo->preprocess(imgsBatch);  float t1 = d_t1.getUsedTime();
	utils::DeviceTimer d_t2; yolo->infer();				  float t2 = d_t2.getUsedTime();
	utils::DeviceTimer d_t3; 
	yolo->setBoxes(yolo->postprocessGetbox(imgsBatch));
    float t3 = d_t3.getUsedTime();

    // 输出提取的边框信息
	cout << " boundingbox information: " << endl;
    for(const auto& box: yolo->getBoxes()){
        std::string class_id;
        float x_lt, y_lt, x_rb, y_rb, confidence;
        std::tie(class_id, x_lt, y_lt, x_rb, y_rb, confidence) = box;
        std::cout << "Class ID: " << class_id << ", Coordinates: (" << x_lt << ", " << y_lt << "), (" << x_rb << ", " << y_rb << ")" << ", onfidence: " << confidence<< std::endl;
    }

	sample::gLogInfo << 
		//"copy time = " << t0 / param.batch_size << "; "
		"preprocess time = " << t1 / param.batch_size << "ms; "
		"infer time = " << t2 / param.batch_size << "ms; "
		"postprocess time = " << t3 / param.batch_size << "ms;" <<std::endl;

	// if(isShow)
	// 	utils::show(yolo.getObjectss(), param.class_names, delayTime, imgsBatch);
	if(isSave)
		// utils::saveIMG(yolo->getBoxes(), param.class_names, param.save_path, imgsBatch, param.batch_size, batchi);
		yolo->saveImage(imgsBatch, yolo->getBoxes());
	yolo->reset();
}

// 优化版本（上传的图片不再储存在服务器，使用变量存储）
int main(int argc, char** argv) {
	try 
	{
		crow::SimpleApp app;

		// 上传文件处理路由
		CROW_ROUTE(app, "/pic_infer").methods(crow::HTTPMethod::Post)([&](const crow::request& req) {
			using json = nlohmann::json;
			json boxes_json = json::array();
			json response_json;

			std::string model_path = "../../weights/yolov8n.trt";  // 默认值
			std::string video_path = ""; 
			std::string image_path = "";
			int camera_id = 0;  
			bool is_save = false;

			std::cout << "Received request, processing multipart data..." << std::endl;
			crow::multipart::message_view file_message(req);

			// 遍历 multipart 中的各部分
			for (const auto& part : file_message.part_map) {
				const auto& part_name = part.first;
				const auto& part_value = part.second;
				if (part_name == "model_path") 
				{
					model_path = std::string(part_value.body.begin(), part_value.body.end());
				} 
				else if (part_name == "video_path") 
				{
					video_path = std::string(part_value.body.begin(), part_value.body.end());
				} 
				else if (part_name == "image_path") 
				{
					image_path = std::string(part_value.body.begin(), part_value.body.end());
				} 
				else if (part_name == "camera_id") 
				{
					camera_id = std::stoi(std::string(part_value.body.begin(), part_value.body.end()));
				} 
				else if (part_name == "is_save") 
				{
					is_save = !part_value.body.empty();
				} 
				else if ("img" == part_name) 
				{
					std::cout << "Received parameters - model_path: " << model_path 
					<< ", video_path: " << video_path 
					<< ", image_path: " << image_path 
					<< ", camera_id: " << camera_id 
					<< ", is_save: " << is_save << std::endl;
					std::cout << "Found 'img', processing..." << std::endl;

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
					std::cout << "File uploaded: " << filename << std::endl;

					// 使用 OpenCV 将上传的文件内容转换为 cv::Mat
					std::vector<uchar> img_data(part_value.body.begin(), part_value.body.end());
					cv::Mat img = cv::imdecode(img_data, cv::IMREAD_COLOR);
					if (img.empty()) {
						std::cout << "Error: Failed to decode image" << std::endl;
						return crow::response(400, "Invalid image data");
					}

					std::cout << "Image decoded successfully, saving to temp file..." << std::endl;

					utils::InitParameter param;
					setParameters(param);
					// 输入流设置(传入 MAT)
					utils::InputStream source;
					source = utils::InputStream::MAT;
					// 更新参数
					bool is_show = false;

					// 初始化输入流
					int total_batches = 0;
					int delay_time = 1;
					cv::VideoCapture capture;
					if (!setInputStream(source, img, video_path, camera_id, capture, total_batches, delay_time, param)) {
						sample::gLogError << "read the input data errors!" << std::endl;
					}
					// 创建新的 YOLOV8 实例（每次请求都创建）
					std::unique_ptr<YOLOV8> yolo = std::make_unique<YOLOV8>(param);
					std::vector<unsigned char> trt_file = utils::loadModel(model_path);
					if (trt_file.empty()) {
						sample::gLogError << "trt_file is empty!" << std::endl;
						return crow::response(500, "Failed to load model");
					}

					if (!yolo->init(trt_file)) {
						sample::gLogError << "initEngine() occurred errors!" << std::endl;
					}
					yolo->check();

					// 处理上传的图像
					try {
						if (img.empty()) {
							sample::gLogError << "Failed to read the image from path: " << image_path << std::endl;
							throw std::runtime_error("Failed to read image");
							return crow::response(400, "Failed to load image");
						}
						else
						{
							std::cout << "Image size: " << img.size() << std::endl;
							std::cout << "Image type: " << img.type() << std::endl;
							std::cout << "Image channels: " << img.channels() << std::endl;
						}
					} catch (const std::exception& e) {
						return crow::response(500, "Error processing image: " + std::string(e.what()));
					}

					std::vector<cv::Mat> imgs_batch;
					imgs_batch.reserve(param.batch_size);
					imgs_batch.emplace_back(img.clone());

					// 获取推理结果并返回
					task(yolo, param, imgs_batch, delay_time, 0, is_show, is_save);
					std::cout << "Detected boxes: " << yolo->getBoxes().size() << std::endl;

					// 构建返回的JSON
					for (const auto& box : yolo->getBoxes()) {
						std::string class_id;
						float x_lt, y_lt, x_rb, y_rb, confidence;
						std::tie(class_id, x_lt, y_lt, x_rb, y_rb, confidence) = box;

						json box_json;
						box_json["box"] = {x_lt, y_lt, x_rb, y_rb};
						box_json["type"] = class_id;
						box_json["confidence"] = confidence;
						boxes_json.push_back(box_json);
					}

					response_json = {
						{"code", 200},
						{"message", "success"},
						{"data", {
							{"json", boxes_json},
							// {"img", base64_str}
						}}
					};
					// cout << "response_json: " << response_json.dump(5) << endl;
					
					img.release();  // 显式释放内存
					capture.release();
					return crow::response(response_json.dump(5));  // 返回推理结果
				}
			}

			std::cout << "Error: No file part named 'img' found in request" << std::endl;
			return crow::response(400);  // 未找到文件部分
		});

		app.port(8866).multithreaded().run();  // 启动 HTTP 服务
	} 
	catch (const std::exception& e) 
	{
		std::cerr << "Caught exception: " << e.what() << std::endl;
		return -1;  // 处理异常，确保程序不会意外退出
	}
	return 0;  // 成功结束
}



// 测试方案1：本地图片测试（用于服务器开发测试）
// int main(int argc, char** argv)
// {
// 	cv::CommandLineParser parser(argc, argv,
// 		{
// 			"{model 	|| tensorrt model file	   }"
// 			"{size      || image (h, w), eg: 640   }"
// 			"{batch_size|| batch size              }"
// 			"{video     || video's path			   }"
// 			"{img       || image's path			   }"
// 			"{cam_id    || camera's device id	   }"
// 			"{show      || if show the result	   }"
// 			"{savePath  || save path, can be ignore}"
// 		});
// 	// parameters
// 	utils::InitParameter param;
// 	setParameters(param);
// 	// path
// 	std::string model_path = "../../yolov8n.trt";
// 	std::string video_path = "../../data/people.mp4";
// 	std::string image_path = "../../data/bus.jpg";
// 	// camera' id
// 	int camera_id = 0;

// 	// get input
// 	utils::InputStream source;
// 	source = utils::InputStream::IMAGE;

// 	// update params from command line parser
// 	int size = -1; // w or h
// 	int batch_size = 8;
// 	bool is_show = false;
// 	bool is_save = false;
// 	if(parser.has("model"))
// 	{
// 		model_path = parser.get<std::string>("model");
// 		sample::gLogInfo << "model_path = " << model_path << std::endl;
// 	}
// 	if(parser.has("size"))
// 	{
// 		size = parser.get<int>("size");
// 		sample::gLogInfo << "size = " << size << std::endl;
// 		param.dst_h = param.dst_w = size;
// 	}
// 	if(parser.has("batch_size"))
// 	{
// 		batch_size = parser.get<int>("batch_size");
// 		sample::gLogInfo << "batch_size = " << batch_size << std::endl;
// 		param.batch_size = batch_size;
// 	}
// 	if(parser.has("img"))
// 	{
// 		source = utils::InputStream::IMAGE;
// 		image_path = parser.get<std::string>("img");
// 		sample::gLogInfo << "image_path = " << image_path << std::endl;
// 	}
// 	if(parser.has("cam_id"))
// 	{
// 		source = utils::InputStream::CAMERA;
// 		camera_id = parser.get<int>("cam_id");
// 		sample::gLogInfo << "camera_id = " << camera_id << std::endl;
// 	}
// 	if(parser.has("show"))
// 	{
// 		is_show = true;
// 		sample::gLogInfo << "is_show = " << is_show << std::endl;
// 	}
// 	if(parser.has("savePath"))
// 	{
// 		is_save = true;
// 		param.save_path = parser.get<std::string>("savePath");
// 		sample::gLogInfo << "save_path = " << param.save_path << std::endl;
// 	}

// 	int total_batches = 0;
// 	int delay_time = 1;
// 	cv::VideoCapture capture;
// 	if (!setInputStream(source, image_path, video_path, camera_id,
// 		capture, total_batches, delay_time, param))
// 	{
// 		sample::gLogError << "read the input data errors!" << std::endl;
// 		return -1;
// 	}

// 	YOLOV8 yolo(param);

// 	// read model
// 	std::vector<unsigned char> trt_file = utils::loadModel(model_path);
// 	if (trt_file.empty())
// 	{
// 		sample::gLogError << "trt_file is empty!" << std::endl;
// 		return -1;
// 	}
// 	// init model
// 	if (!yolo.init(trt_file))
// 	{
// 		sample::gLogError << "initEngine() ocur errors!" << std::endl;
// 		return -1;
// 	}
// 	yolo.check();
// 	cv::Mat frame;
// 	std::vector<cv::Mat> imgs_batch;
// 	imgs_batch.reserve(param.batch_size);
// 	sample::gLogInfo << imgs_batch.capacity() << std::endl;
// 	int batchi = 0;


// 	crow::SimpleApp app;
//     // 首页欢迎信息 
//     CROW_ROUTE(app, "/")([]() {
//         return R"({"message": "Hello World"})";
//     });

// 	CROW_ROUTE(app, "/pic_infer").methods("GET"_method, "POST"_method)(
// 		[&](const crow::request& req) {
// 			// json 的封装
// 			using json = nlohmann::json;
// 			json boxes_json = json::array();
// 			json response_json;

// 			while (capture.isOpened()) {
// 				if (batchi >= total_batches && source != utils::InputStream::CAMERA) {
// 					break;
// 				}
// 				if (imgs_batch.size() < param.batch_size) { // get input
// 					if (source != utils::InputStream::IMAGE) {
// 						capture.read(frame);
// 					} else {
// 						frame = cv::imread(image_path);
// 					}

// 					imgs_batch.emplace_back(frame.clone());
// 				} else {
// 					std::cout << "Entering task function..." << std::endl;
// 					std::vector<std::tuple<int, float, float, float, float>> detected_boxes;
// 					task(yolo, param, imgs_batch, delay_time, batchi, is_show, is_save, detected_boxes);

// 					if (detected_boxes.empty()) {
// 						std::cout << "No bounding boxes returned." << std::endl;
// 					}
// 					// 输出提取的边框信息
// 					std::cout << "boundingbox information: " << std::endl;
// 					for (const auto& box : detected_boxes) {
// 						int class_id;
// 						float x_lt, y_lt, x_rb, y_rb;
// 						std::tie(class_id, x_lt, y_lt, x_rb, y_rb) = box;
// 						std::cout << "Class ID: " << class_id << ", Coordinates: ("
// 								<< x_lt << ", " << y_lt << "), (" << x_rb << ", " << y_rb << ")"
// 								<< std::endl;
// 						std::cout.flush();  // 强制刷新输出
//                         // 为每个框创建一个单独的 JSON 对象并添加到 boxes_json_array
//                         json box_json;
// 						box_json["box"] = {x_lt, y_lt, x_rb, y_rb};
// 						box_json["type"] = class_id;
// 						boxes_json.push_back(box_json);
// 					}

// 					response_json = {
// 						{"code", 200},
// 						{"message", "success"},
// 						{"data",{
// 							{"json", boxes_json},
// 							// {"img", base64_str}
// 						}}
// 					};

// 					// 你可以选择将 response_json 返回或存储，以下为例：
//                 	// std::cout << "Response JSON: " << response_json.dump(4) << std::endl;

// 					imgs_batch.clear();
// 					batchi++;
// 				}
// 			}

// 			// 返回一个成功的响应（可以根据需要调整响应内容）
// 			return crow::response(response_json.dump(4));
// 		}
// 	);

// 	app.port(18080).run();  // 不使用多线程模式
// }



// 测试方案2：处理用户上传的图片
// std::string generateTempFilename(const std::string& extension) {
//     // 使用当前时间戳生成唯一的临时文件名
//     std::time_t now = std::time(0);
//     char temp_filename[100];
//     std::strftime(temp_filename, sizeof(temp_filename), "%Y%m%d%H%M%S", std::localtime(&now));
//     return std::string(temp_filename) + extension;  // 使用传入的扩展名
// }

// int main(int argc, char** argv) {
// 	try 
// 	{
// 		crow::SimpleApp app;

// 		// 上传文件处理路由
// 		CROW_ROUTE(app, "/pic_infer").methods(crow::HTTPMethod::Post)([&](const crow::request& req) {
// 			using json = nlohmann::json;
// 			json boxes_json = json::array();
// 			json response_json;

// 			std::string model_path = "../../yolov8n.trt";  // 默认值
// 			std::string video_path = "";  // 默认值
// 			std::string image_path = "";
// 			int camera_id = 0;  // 默认值
// 			bool is_save = false;

// 			std::cout << "Received request, processing multipart data..." << std::endl;
// 			crow::multipart::message_view file_message(req);

// 			// 遍历 multipart 中的各部分
// 			for (const auto& part : file_message.part_map) {
// 				const auto& part_name = part.first;
// 				const auto& part_value = part.second;
// 				if (part_name == "model_path") {
// 					model_path = std::string(part_value.body.begin(), part_value.body.end());
// 				} else if (part_name == "video_path") {
// 					video_path = std::string(part_value.body.begin(), part_value.body.end());
// 				} else if (part_name == "image_path") {
// 					image_path = std::string(part_value.body.begin(), part_value.body.end());
// 				} else if (part_name == "camera_id") {
// 					camera_id = std::stoi(std::string(part_value.body.begin(), part_value.body.end()));
// 				} else if (part_name == "is_save") {
// 					is_save = !part_value.body.empty();
// 				} else if ("img" == part_name) {
// 					std::cout << "Received parameters - model_path: " << model_path 
// 					<< ", video_path: " << video_path 
// 					<< ", image_path: " << image_path 
// 					<< ", camera_id: " << camera_id 
// 					<< ", is_save: " << is_save << std::endl;
// 					std::cout << "Found 'img', processing..." << std::endl;

// 					// 从请求头中获取文件名
// 					auto headers_it = part_value.headers.find("Content-Disposition");
// 					if (headers_it == part_value.headers.end()) {
// 						std::cout << "Error: Missing 'Content-Disposition' header" << std::endl;
// 						return crow::response(400);  // 请求头中没有 Content-Disposition
// 					}
// 					auto params_it = headers_it->second.params.find("filename");
// 					if (params_it == headers_it->second.params.end()) {
// 						std::cout << "Error: Missing filename in 'Content-Disposition' header" << std::endl;
// 						return crow::response(400);  // 没有文件名
// 					}
// 					const std::string filename(params_it->second);
// 					std::cout << "File uploaded: " << filename << std::endl;

// 					// 使用 OpenCV 将上传的文件内容转换为 cv::Mat
// 					std::vector<uchar> img_data(part_value.body.begin(), part_value.body.end());
// 					cv::Mat img = cv::imdecode(img_data, cv::IMREAD_COLOR);
// 					if (img.empty()) {
// 						std::cout << "Error: Failed to decode image" << std::endl;
// 						return crow::response(400, "Invalid image data");
// 					}

// 					std::cout << "Image decoded successfully, saving to temp file..." << std::endl;

// 					// 将上传的图像保存为临时文件
// 					std::string temp_image_path = generateTempFilename(".jpg");
// 					if (!cv::imwrite(temp_image_path, img)) {
// 						std::cout << "Error: Failed to save image to temporary file" << std::endl;
// 						return crow::response(500, "Failed to save image");
// 					}

// 					std::cout << "Image saved to temporary file: " << temp_image_path << std::endl;

// 					utils::InitParameter param;
// 					setParameters(param);
// 					image_path = temp_image_path;
// 					// 输入流设置
// 					utils::InputStream source;
// 					source = utils::InputStream::IMAGE;

// 					// 更新参数
// 					bool is_show = false;

// 					// 初始化输入流
// 					int total_batches = 0;
// 					int delay_time = 1;
// 					cv::VideoCapture capture;
// 					if (!setInputStream(source, image_path, video_path, camera_id, capture, total_batches, delay_time, param)) {
// 						sample::gLogError << "read the input data errors!" << std::endl;
// 						// return -1;
// 					}
// 					// 创建新的 YOLOV8 实例（每次请求都创建）
// 					std::unique_ptr<YOLOV8> yolo = std::make_unique<YOLOV8>(param);
// 					std::vector<unsigned char> trt_file = utils::loadModel(model_path);
// 					if (trt_file.empty()) {
// 						sample::gLogError << "trt_file is empty!" << std::endl;
// 						return crow::response(500, "Failed to load model");
// 						// return -1;
// 					}

// 					if (!yolo->init(trt_file)) {
// 						sample::gLogError << "initEngine() occurred errors!" << std::endl;
// 						// return -1;
// 					}
// 					yolo->check();

// 					cv::Mat frame;
// 					// 处理上传的图像
// 					try {
// 						frame = cv::imread(image_path);  // 将 frame 作为引用传递
// 						if (frame.empty()) {
// 							sample::gLogError << "Failed to read the image from path: " << image_path << std::endl;
// 							throw std::runtime_error("Failed to read image");
// 							return crow::response(400, "Failed to load image");
// 						}
// 						else
// 						{
// 							cout << "重新读取成功: "<< image_path << endl;
// 							std::cout << "Image size: " << frame.size() << std::endl;
// 							std::cout << "Image type: " << frame.type() << std::endl;
// 							std::cout << "Image channels: " << frame.channels() << std::endl;
// 						}
// 					} catch (const std::exception& e) {
// 						return crow::response(500, "Error processing image: " + std::string(e.what()));
// 					}

// 					std::vector<cv::Mat> imgs_batch;
// 					imgs_batch.reserve(param.batch_size);
// 					imgs_batch.emplace_back(frame.clone());

// 					// 获取推理结果并返回
// 					// std::vector<std::tuple<std::string, float, float, float, float>> detected_boxes;
// 					task(yolo, param, imgs_batch, delay_time, 0, is_show, is_save);
// 					// std::cout << "Detected boxes: " << detected_boxes.size() << std::endl;
// 					std::cout << "Detected boxes: " << yolo->getBoxes().size() << std::endl;

// 					// 构建返回的JSON
// 					for (const auto& box : yolo->getBoxes()) {
// 						std::string class_id;
// 						float x_lt, y_lt, x_rb, y_rb, confidence;
// 						std::tie(class_id, x_lt, y_lt, x_rb, y_rb, confidence) = box;

// 						json box_json;
// 						box_json["box"] = {x_lt, y_lt, x_rb, y_rb};
// 						box_json["type"] = class_id;
// 						box_json["confidence"] = confidence;
// 						boxes_json.push_back(box_json);
// 					}

// 					response_json = {
// 						{"code", 200},
// 						{"message", "success"},
// 						{"data", {
// 							{"json", boxes_json},
// 							// {"img", base64_str}
// 						}}
// 					};
// 					// cout << "response_json: " << response_json.dump(5) << endl;
					
// 					if (std::remove(temp_image_path.c_str()) != 0) {
// 						std::cerr << "Error deleting temporary file: " << temp_image_path << std::endl;
// 					} else {
// 						std::cout << "Temporary file deleted successfully: " << temp_image_path << std::endl;
// 					}
// 					frame.release();  // 显式释放内存
// 					capture.release();
// 					return crow::response(response_json.dump(5));  // 返回推理结果
// 				}
// 			}

// 			std::cout << "Error: No file part named 'img' found in request" << std::endl;
// 			return crow::response(400);  // 未找到文件部分
// 		});

// 		app.port(8866).run();  // 启动 HTTP 服务
// 	} 
// 	catch (const std::exception& e) 
// 	{
// 		std::cerr << "Caught exception: " << e.what() << std::endl;
// 		return -1;  // 处理异常，确保程序不会意外退出
// 	}
// 	return 0;  // 成功结束
// }