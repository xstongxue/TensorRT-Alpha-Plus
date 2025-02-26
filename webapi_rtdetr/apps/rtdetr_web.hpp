#ifndef LINFER_RTDETRWEB_HPP
#define LINFER_RTDETRWEB_HPP

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>
#include <string_view>
#include <array>
#include <mutex>

// 定义 inline 变量，确保在多个源文件中共享
// inline const std::array<std::string_view, 80> cocolabels = {
//     "question", "illu", "car", "motorcycle", "airplane",
//     "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
//     "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
//     "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
//     "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
//     "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
//     "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
//     "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
//     "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
//     "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
//     "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
//     "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
//     "scissors", "teddy bear", "hair drier", "toothbrush"
// };

// 测试公司版面分析模型
inline const std::array<std::string_view, 80> cocolabels = {
    "background", "illu", "question", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};


/// -------------------------- 封装接口类 ---------------------------
namespace RTDETRWEB{

    using namespace std;

    struct Box{
        float left, top, right, bottom, confidence;
        int label;

        Box() = default;

        Box(float left, float top, float right, float bottom, float confidence, int label)
                :left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label){}
    };

    using BoxArray = std::vector<Box>;

    class Detect{
    public:
        virtual shared_future<BoxArray> commit_getBoxes(const cv::Mat& image) = 0;
        virtual vector<shared_future<BoxArray>> commits_getBoxes(const vector<cv::Mat>& images) = 0;
    };

    shared_ptr<Detect> create_infer(
            const string& engine_file, int gpuid,
            float confidence_threshold = 0.4f, int max_objects = 300,
            bool use_multi_preprocess_stream = false
    );


} // namespace RTDETRWEB


#endif //LINFER_RTDETRWEB_HPP
