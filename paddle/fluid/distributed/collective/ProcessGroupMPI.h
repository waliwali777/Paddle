// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <mpi.h>
#include <chrono>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>

#include "paddle/fluid/distributed/collective/ProcessGroup.h"
#include "paddle/fluid/distributed/collective/Types.h"
#include "paddle/fluid/platform/device_context.h"

constexpr const char* MPI_BACKEND_NAME = "MPI";

namespace paddle {
namespace distributed {

struct TaskEntry {
  explicit TaskEntry(std::vector<phi::DenseTensor>* src_ptr,
                     std::vector<phi::DenseTensor>* dst_ptr,
                     std::function<void(std::unique_ptr<TaskEntry>&)> run)
      : dst(dst_ptr ? *dst_ptr : std::vector<phi::DenseTensor>()),
        run(std::move(run)) {
    if (src_ptr) {
      src = *src_ptr;
    }
  }

  TaskEntry(const TaskEntry&) = delete;
  TaskEntry& operator=(const TaskEntry&) = delete;

  // For input and output tensors (in-place), we will always use src
  std::vector<phi::DenseTensor> src;

  // Copy of user provided outputs.
  const std::vector<phi::DenseTensor> dst;

  // src rank returned, for recv only
  int* srcRank = nullptr;
  std::function<void(std::unique_ptr<TaskEntry>&)> run;
};

class ProcessGroupMPI : public ProcessGroup {
 public:
  class MPITask : public ProcessGroup::Task {
   public:
    explicit MPITask(std::vector<phi::DenseTensor> outputTensors,
                     const std::vector<phi::DenseTensor>& inputTensors)
        : ProcessGroup::Task(-1, inputTensors, CommType::UNKNOWN),
          outputTensors_(std::move(outputTensors)) {}

    void synchronize() {}

    bool wait(std::chrono::milliseconds timeout = kWaitTimeout) {
      std::unique_lock<std::mutex> lock(mutex_);
      if (timeout == kWaitTimeout) {
        // This waits without a timeout.
        cv_.wait(lock, [&] { return is_completed_; });
      } else {
        // Waits for the user-provided timeout.
        cv_.wait_for(lock, timeout, [&] { return is_completed_; });
        PADDLE_ENFORCE_EQ(
            is_completed_, true,
            platform::errors::InvalidArgument("MPI operation timeout! "));
      }
      if (exception_) {
        std::rethrow_exception(exception_);
      }
      return true;
    }

   protected:
    friend class ProcessGroupMPI;

   private:
    // about mpi
    void finish(std::exception_ptr exception) {
      is_completed_ = true;
      exception_ = exception;
      cv_.notify_all();
    }
    void finishMPITask();
    void finishMPITaskError(std::exception_ptr eptr);

    std::vector<phi::DenseTensor> outputTensors_;
    std::condition_variable cv_;
    std::exception_ptr exception_;
  };

 public:
  class MPIAsyncTask : public ProcessGroup::Task {
   public:
    MPIAsyncTask(MPI_Request request,
                 const std::vector<phi::DenseTensor>& inputs);

    bool IsCompleted();

    void Synchronize() {}

    bool Wait(std::chrono::milliseconds timeout = kWaitTimeout);

    void SetOutputs(std::vector<phi::DenseTensor>& outputs);  // NOLINT

    virtual ~MPIAsyncTask();

   protected:
    void appearException();

   private:
    std::shared_ptr<std::vector<phi::DenseTensor>> outputs_;
    MPI_Request request_;
    MPI_Status status_;
  };

  ProcessGroupMPI(int rank, int size, MPI_Comm pgComm, int gid);
  ~ProcessGroupMPI();

  const std::string GetBackendName() const override {
    return std::string(MPI_BACKEND_NAME);
  }

  // std::shared_ptr<ProcessGroup::Task> AllReduce(
  //    std::vector<phi::DenseTensor>& in_tensors,
  //    std::vector<phi::DenseTensor>& out_tensors,
  //    const AllreduceOptions& = AllreduceOptions()) override;

  std::shared_ptr<ProcessGroup::Task> Broadcast(
      std::vector<phi::DenseTensor>& in_tensors,
      std::vector<phi::DenseTensor>& out_tensors,
      const BroadcastOptions& = BroadcastOptions()) override;

  // std::shared_ptr<ProcessGroup::Task> Barrier(
  //    const BarrierOptions& = BarrierOptions()) override;

  // std::shared_ptr<ProcessGroup::Task> Send(
  //    std::vector<phi::DenseTensor>& tensors, int dst_rank) override;

  // std::shared_ptr<ProcessGroup::Task> Recv(
  //    std::vector<phi::DenseTensor>& tensors, int src_rank) override;

  // std::shared_ptr<ProcessGroup::Task> AllGather(
  //    std::vector<phi::DenseTensor>& in_tensors,
  //    std::vector<phi::DenseTensor>& out_tensors) override;

  // std::shared_ptr<ProcessGroup::Task> AllToAll(
  //    std::vector<phi::DenseTensor>& in,
  //    std::vector<phi::DenseTensor>& out) override;

  // std::shared_ptr<ProcessGroup::Task> Reduce(
  //    std::vector<phi::DenseTensor>& tensors,
  //    std::vector<phi::DenseTensor>& out_tensors,
  //    const ReduceOptions& opts) override;

  // std::shared_ptr<ProcessGroup::Task> Scatter(
  //    std::vector<phi::DenseTensor>& in_tensors,
  //    std::vector<phi::DenseTensor>& out_tensors,
  //    const ScatterOptions&) override;

  static std::shared_ptr<ProcessGroupMPI> createProcessGroupMPI(
      std::vector<int> ranks, int gid);

 protected:
  // Worker thread loop
  void workLoop();

  std::shared_ptr<ProcessGroup::Task> enqueue(
      std::unique_ptr<TaskEntry> entry,
      const std::vector<phi::DenseTensor>& inputs);

 private:
  bool stop_;
  std::mutex pg_mutex;

  // main dispatch thread
  std::thread worker_thread;

  std::deque<std::tuple<std::unique_ptr<TaskEntry>, std::shared_ptr<MPITask>>>
      queue_;
  std::condition_variable queue_produce;
  std::condition_variable queue_consume;

  // Global states
  static void initOneTimeMPI();
  static void exitMPI();
  static std::once_flag onceFlag;

  static std::mutex pg_global_mutex;
  // static int mpi_thread_support;

  MPI_Comm pg_comm;
};

}  //  namespace distributed
}  //  namespace paddle
