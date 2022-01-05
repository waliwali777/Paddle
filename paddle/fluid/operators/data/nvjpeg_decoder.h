/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <vector>
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/dynload/nvjpeg.h"
#include "paddle/fluid/platform/stream/cuda_stream.h"

#include "paddle/fluid/operators/data/random_roi_generator.h"

namespace paddle {
namespace operators {
namespace data {

static int dev_malloc(void **p, size_t s) { return (int)cudaMalloc(p, s); }
static int dev_free(void *p) { return (int)cudaFree(p);  }

static int host_malloc(void** p, size_t s, unsigned int f) { return (int)cudaHostAlloc(p, s, f); }
static int host_free(void* p) { return (int)cudaFreeHost(p);  }

struct NvjpegDecodeTask {
  const uint8_t* bit_stream;
  size_t bit_len;
  framework::LoDTensor* tensor;
  RandomROIGenerator* roi_generator;
  platform::Place place;
};

class NvjpegDecoder {
  public:
    NvjpegDecoder(const std::string mode, int dev_id);

    ~NvjpegDecoder();

    void Run(const uint8_t* bit_stream, size_t bit_len, framework::LoDTensor* out, 
             RandomROIGenerator* roi_generator, platform::Place& place);

  private:
    DISABLE_COPY_AND_ASSIGN(NvjpegDecoder);

    void ParseDecodeParams(
        const uint8_t* bit_stream, size_t bit_len, framework::LoDTensor* out,
        RandomROIGenerator* roi_generator, nvjpegImage_t* out_image,
        platform::Place place);

    void Decode(const uint8_t* bit_stream, size_t bit_len, nvjpegImage_t* out_image);


    cudaStream_t cuda_stream_ = nullptr;
    std::vector<nvjpegJpegStream_t> nvjpeg_streams_;

    nvjpegHandle_t handle_ = nullptr;
    nvjpegJpegState_t state_ = nullptr;
    nvjpegJpegDecoder_t decoder_ = nullptr;
    nvjpegDecodeParams_t decode_params_ = nullptr;

    nvjpegPinnedAllocator_t pinned_allocator_ = {&host_malloc, &host_free};
    nvjpegDevAllocator_t device_allocator_ = {&dev_malloc, &dev_free};
    std::vector<nvjpegBufferPinned_t> pinned_buffers_;
    nvjpegBufferDevice_t device_buffer_ = nullptr;

    int page_id_;

    const std::string mode_;
};

class NvjpegDecoderThreadPool {
  public:
    NvjpegDecoderThreadPool(const int num_threads, const std::string mode, const int dev_id);

    ~NvjpegDecoderThreadPool();

    void AddTask(std::shared_ptr<NvjpegDecodeTask> task);

    void RunAll(const bool wait, const bool sort = true);

    void WaitTillTasksCompleted();

    void ShutDown();

  private:
    DISABLE_COPY_AND_ASSIGN(NvjpegDecoderThreadPool);

    void SortTaskByLengthDescend();

    void ThreadLoop(const int thread_idx);

    std::vector<std::thread> threads_;
    std::string mode_;
    int dev_id_;

    std::deque<std::shared_ptr<NvjpegDecodeTask>> task_queue_;
    std::mutex mutex_;

    bool shutdown_;
    std::condition_variable running_cond_;
    bool running_;
    std::condition_variable completed_cond_;
    bool completed_;

    int outstand_tasks_;
};

}  // namespace data
}  // namespace operators
}  // namespace paddle
