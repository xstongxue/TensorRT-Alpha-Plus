#ifndef LINFER_RTDETRWEB_HPP
#define LINFER_RTDETRWEB_HPP

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>


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
