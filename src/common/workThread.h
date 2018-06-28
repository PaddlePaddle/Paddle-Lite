#pragma once

#include <thread>
#include <condition_variable>
#include <mutex>
#include <iostream>
// #include <function>
// #include <queue>

template <typename T>
class workThread {
public:
    void init(std::vector<T> & ops) {
        this->ops = ops;
        t = std::thread([this]{
            threadLoop();
        });
    }
    void threadLoop() {
        //std::unique_lock<std::mutex> lk(m);
        for (;;) {
            while(true) {
                std::lock_guard<std::mutex> lg(m);
                if (job != -1) break;
            };
            // std::cout << "loop!" << "\n";
            // cv.wait(lk, [this]{
            //     // std::cout << "wake up!!! you have work to do: " << job << "\n";
            //     return job != -1;
            // });
            if (job == -2) {
                break;
            }
            // TODO run job
            ops[job]->Run();
            job = -1;
            cv.notify_one();
        }
    }
    void run(int i) {
        std::lock_guard<std::mutex> lk(m);
        job = i;
        cv.notify_one();
    }
    void join() {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [this]{
            return job == -1;
        });
    }
    void quit() {
        {
            std::cout << "try get lock!!!\n";
            std::lock_guard<std::mutex> lk(m);
            std::cout << "set job = -2\n";
            job = -2;
            cv.notify_one();
        }
        t.join();
        std::cout << "job quit\n";
    }
private:
    std::vector<T> ops;
    // std::queue<std::function<void()>> q;
    int job = -1;
    // std::atomic<int> job;
    std::condition_variable cv;
    std::mutex m;
    std::thread t;
    // bool idle = false;
};
