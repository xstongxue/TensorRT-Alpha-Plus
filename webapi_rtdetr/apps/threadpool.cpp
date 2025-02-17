#include "threadpool.hpp"

ThreadPool::ThreadPool(size_t num_threads) : stop(false) {
    for (size_t i = 0; i < num_threads; ++i) {
        workers.emplace_back(std::make_unique<std::thread>(&ThreadPool::workerThread, this));
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queMtx);
        stop = true;
    }
    queCond.notify_all();
    for (auto& worker : workers) {
        worker->join();
    }
}

std::future<void> ThreadPool::submitTask(std::shared_ptr<Task> task) {
    return std::async(std::launch::async, &Task::execute, task);
}

void ThreadPool::workerThread() {
    while (true) {
        std::shared_ptr<Task> task = nullptr;
        {
            std::unique_lock<std::mutex> lock(queMtx);
            queCond.wait(lock, [this] { return stop || !tasks.empty(); });
            if (stop && tasks.empty()) return;
            task = tasks.front();
            tasks.pop();
        }
        if (task) {
            task->execute();
        }
    }
}
