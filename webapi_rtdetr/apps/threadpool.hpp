#pragma once
#include <iostream>
#include <vector>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <queue>
#include <atomic>
#include <memory>
#include <future>

class Task {
public:
    virtual void execute() = 0;
    virtual ~Task() = default;  // 纯虚函数，要求子类实现
};

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads);
    ~ThreadPool();

    std::future<void> submitTask(std::shared_ptr<Task> task);

private:
    std::atomic<bool> stop;
    std::mutex queMtx;
    std::condition_variable queCond;
    std::vector<std::unique_ptr<std::thread>> workers;
    std::queue<std::shared_ptr<Task>> tasks;

    void workerThread();
};
