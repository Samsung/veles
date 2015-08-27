/*
Copyright © 2012 Jakob Progsch, Václav Zeman, 2015 Vadim Markovtsev

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

   1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.

   2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.

   3. This notice may not be removed or altered from any source
   distribution.
*/

#ifndef SRC_THREAD_POOL_H
#define SRC_THREAD_POOL_H

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace veles {

namespace internal {

/// Executes concurrent tasks on a fixed number of threads.
class ThreadPool {
 public:
  /// Constructor launches the requested number of workers.
  explicit ThreadPool(
      size_t threads_number = std::thread::hardware_concurrency())
      : stop_(false) {
    assert(threads_number && "There must be 1 or more threads");
    workers_.reserve(threads_number);
    for(; threads_number; --threads_number) {
      workers_.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this]{ return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) {
              return;
            }
            task = std::move(tasks_.front());
            tasks_.pop();
          }
          task();
        }
      });
    }
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  /// Enqueues the task.
  template<class F, class... Args>
  std::shared_future<typename std::result_of<F(Args...)>::type>
  enqueue(F&& f, Args&&... args) {
    using packaged_task_t =
        std::packaged_task<typename std::result_of<F(Args...)>::type()>;
    using shared_future_t =
        std::shared_future<typename std::result_of<F(Args...)>::type>;

    std::shared_ptr<packaged_task_t> task(new packaged_task_t(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    ));
    auto res = shared_future_t(task->get_future());
    {
      std::unique_lock<std::mutex> lock(mutex_);
      tasks_.emplace([task]{ (*task)(); });
    }
    condition_.notify_one();
    return std::move(res);
  }

  virtual ~ThreadPool() {
    stop_ = true;
    condition_.notify_all();
    for (auto& worker : workers_) {
      worker.join();
    }
  }

 private:
  /// need to keep track of threads so we can join them
  std::vector< std::thread > workers_;
  /// the task queue
  std::queue< std::function<void()> > tasks_;
  /// synchronization
  std::mutex mutex_;
  std::condition_variable condition_;
  /// workers finalization flag
  std::atomic_bool stop_;
};

}  // namespace internal

}  // namespace veles

#endif  // SRC_THREAD_POOL_H
