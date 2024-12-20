
#include <opencv2/opencv.hpp>
#include "../rtdetr/apps/app_rtdetr.cpp"

using namespace std;

void performance(std::string img_root, const string& engine_file, int gpuid);
void batch_inference(const string& img_root, const string& engine_file, int gpuid);
void single_inference(const std::string& img_root, const string& engine_file, int gpuid);

void test_rtdetr(std::string img_root, std::string model_root){
//    batch_inference(img_root, model_root, 0);
   single_inference(img_root, model_root, 0);
    // performance(img_root, model_root, 0);
}

int main(){
    std::string img_root = "/data02/xs/code/tensorrt-alpha/rtdetr/";
    std::string model_root = "/data02/xs/code/TensorRT_deploy/RT_DETR_TensorRT/rtdetr-l.trt";
    test_rtdetr(img_root, model_root);
    return 0;
}