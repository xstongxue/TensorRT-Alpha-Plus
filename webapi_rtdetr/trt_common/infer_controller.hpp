#ifndef INFER_CONTROLLER_HPP
#define INFER_CONTROLLER_HPP

/// Producer Consumer Model
/// 提供 InferController 模板类，让具体应用继承
/// 避免重复书写生产者消费者的代码

#include <string>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include "tensor_allocator.hpp"

template<class Input, class Output, class StartParam=std::tuple<std::string, int>, class JobAdditional=int>
class InferControllerWEB{
public:
    struct Job{
        Input input;
        Output output;
        JobAdditional additional;
        TensorAllocator::MonoDataPtr mono_tensor;
        std::shared_ptr<std::promise<Output>> pro;
    };

    virtual ~InferControllerWEB(){
        stop();
    }

    void stop(){
        run_ = false;
        cond_.notify_all();

        // clean jobs
        {
            std::unique_lock<std::mutex> l(jobs_lock_);
            while(!jobs_.empty()){
                auto& item = jobs_.front();
                if(item.pro)
                    item.pro->set_value(Output());
                jobs_.pop();
            }
        }

        if(worker_){
            worker_->join();
            worker_.reset();
        }
    }

    bool startup(const StartParam& param){
        run_ = true;

        std::promise<bool> pro;
        start_param_ = param;
        worker_.reset(new std::thread(&InferControllerWEB::worker, this, std::ref(pro)));
        return pro.get_future().get();
    }

    virtual std::shared_future<Output> commit_getBoxes(const Input& input){

        Job job;
        job.pro = std::make_shared<std::promise<Output>>();
        if(!preprocess(job, input)){
            job.pro->set_value(Output());
            return job.pro->get_future();
        }

        {
            std::unique_lock<std::mutex> l(jobs_lock_);
            jobs_.push(job);
        };
        cond_.notify_one();
        return job.pro->get_future();
    }

    virtual std::vector<std::shared_future<Output>> commits_getBoxes(const std::vector<Input>& inputs){

        int batch_size = std::min((int)inputs.size(), this->tensor_allocator_->capacity());
        std::vector<Job> jobs(inputs.size());
        std::vector<std::shared_future<Output>> results(inputs.size());

        int nepoch = (inputs.size() + batch_size - 1) / batch_size;
        for(int epoch = 0; epoch < nepoch; ++epoch){
            int begin = epoch * batch_size;
            int end   = std::min((int)inputs.size(), begin + batch_size);

            for(int i = begin; i < end; ++i){
                Job& job = jobs[i];
                job.pro = std::make_shared<std::promise<Output>>();
                if(!preprocess(job, inputs[i])){
                    job.pro->set_value(Output{});
                }
                results[i] = job.pro->get_future();
            }

            {
                std::unique_lock<std::mutex> l(jobs_lock_);
                for(int i = begin; i < end; ++i){
                    jobs_.emplace(std::move(jobs[i]));
                };
            }
            cond_.notify_one();
        }
        return results;
    }

protected:
    virtual void worker(std::promise<bool>& result) = 0;
    virtual bool preprocess(Job& job, const Input& input) = 0;
    
    virtual bool get_jobs_and_wait(std::vector<Job>& fetch_jobs, int max_size){

        std::unique_lock<std::mutex> l(jobs_lock_);
        cond_.wait(l, [&](){
            return !run_ || !jobs_.empty();
        });

        if(!run_) return false;
        
        fetch_jobs.clear();
        for(int i = 0; i < max_size && !jobs_.empty(); ++i){
            fetch_jobs.emplace_back(std::move(jobs_.front()));
            jobs_.pop();
        }
        return true;
    }

    virtual bool get_job_and_wait(Job& fetch_job){

        std::unique_lock<std::mutex> l(jobs_lock_);
        cond_.wait(l, [&](){
            return !run_ || !jobs_.empty();
        });

        if(!run_) return false;
        
        fetch_job = std::move(jobs_.front());
        jobs_.pop();
        return true;
    }

protected:
    StartParam start_param_;
    std::atomic<bool> run_{};
    std::mutex jobs_lock_;  // 确保往队列里添加任务和取任务同一时间只能执行一种
    std::queue<Job> jobs_;
    std::unique_ptr<std::thread> worker_;
    std::condition_variable cond_;
    std::unique_ptr<TensorAllocator> tensor_allocator_;
};

#endif // INFER_CONTROLLER_HPP