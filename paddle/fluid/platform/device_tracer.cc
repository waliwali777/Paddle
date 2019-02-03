/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/platform/device_tracer.h"

#include <deque>
#include <forward_list>
#include <fstream>
#include <list>
#include <map>
#include <mutex>  // NOLINT
#include <numeric>
#include <sstream>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/platform/cupti_cbid_str.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/string/printf.h"

namespace paddle {
namespace platform {
namespace {
// Tracking the nested block stacks of each thread.
thread_local std::deque<int> block_id_stack;
// Tracking the nested event stacks.
thread_local std::deque<Event *> annotation_stack;

std::map<uint32_t, int32_t> system_thread_id_map;

std::once_flag tracer_once_flag;
DeviceTracer *tracer = nullptr;
}  // namespace
#ifdef PADDLE_WITH_CUPTI

namespace {
// TODO(panyx0718): Revisit the buffer size here.
uint64_t kBufSize = 32 * 1024;
uint64_t kAlignSize = 8;

#define ALIGN_BUFFER(buffer, align)                                 \
  (((uintptr_t)(buffer) & ((align)-1))                              \
       ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1))) \
       : (buffer))

#define CUPTI_CALL(call)                                                   \
  do {                                                                     \
    CUptiResult _status = call;                                            \
    if (_status != CUPTI_SUCCESS) {                                        \
      const char *errstr;                                                  \
      dynload::cuptiGetResultString(_status, &errstr);                     \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
              __FILE__, __LINE__, #call, errstr);                          \
      exit(-1);                                                            \
    }                                                                      \
  } while (0)

std::string MemcpyKind(CUpti_ActivityMemcpyKind kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
      return "MEMCPY_HtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
      return "MEMCPY_DtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
      return "MEMCPY_HtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
      return "MEMCPY_AtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
      return "MEMCPY_AtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
      return "MEMCPY_AtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
      return "MEMCPY_DtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
      return "MEMCPY_DtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
      return "MEMCPY_HtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
      return "MEMCPY_PtoP";
    case CUPTI_ACTIVITY_MEMCPY_KIND_FORCE_INT:
      return "MEMCPY_FORCE_INT";
    default:
      break;
  }
  return "MEMCPY";
}

std::string DriverKind(CUpti_CallbackId cbid) {
  if (static_cast<size_t>(cbid) >= driver_cbid_str.size())
    return "Driver API " + std::to_string(cbid);
  // remove the "CUPTI_DRIVER_TRACE_CBID_" prefix
  return driver_cbid_str[cbid].substr(24);
}

std::string RuntimeKind(CUpti_CallbackId cbid) {
  if (static_cast<size_t>(cbid) >= runtime_cbid_str.size())
    return "Runtime API " + std::to_string(cbid);
  // remove the "CUPTI_RUNTIME_TRACE_CBID_" prefix
  return runtime_cbid_str[cbid].substr(25);
}

void EnableActivity() {
  // Device activity record is created when CUDA initializes, so we
  // want to enable it before cuInit() or any CUDA runtime call.
  CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(
      dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  // CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
  CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  // We don't track these activities for now.
  // CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
  // CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD));
  // CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));
  // CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
  // CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  // CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  // CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME));
  // CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER));
}

void DisableActivity() {
  CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(
      dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DEVICE));
  // Disable all other activity record kinds.
  CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONTEXT));
  CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));
  CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_NAME));
  CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MARKER));
  CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_OVERHEAD));
}

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size,
                              size_t *maxNumRecords) {
  uint8_t *buf = reinterpret_cast<uint8_t *>(malloc(kBufSize + kAlignSize));
  *size = kBufSize;
  *buffer = ALIGN_BUFFER(buf, kAlignSize);
  *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer,
                              size_t size, size_t validSize) {
  static std::thread::id cupti_thread_id(0);
  if (cupti_thread_id == std::thread::id(0))
    cupti_thread_id = std::this_thread::get_id();
  PADDLE_ENFORCE_EQ(std::this_thread::get_id(), cupti_thread_id,
                    "Only one thread is allowed to call bufferCompleted()");
  CUptiResult status;
  CUpti_Activity *record = NULL;
  if (validSize > 0) {
    do {
      status = dynload::cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        switch (record->kind) {
          case CUPTI_ACTIVITY_KIND_KERNEL:
          case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
            auto *kernel =
                reinterpret_cast<const CUpti_ActivityKernel3 *>(record);
            tracer->AddKernelRecords(kernel->name, kernel->start, kernel->end,
                                     kernel->deviceId, kernel->streamId,
                                     kernel->correlationId);
            break;
          }
          case CUPTI_ACTIVITY_KIND_MEMCPY: {
            auto *memcpy =
                reinterpret_cast<const CUpti_ActivityMemcpy *>(record);
            tracer->AddMemRecords(
                MemcpyKind(
                    static_cast<CUpti_ActivityMemcpyKind>(memcpy->copyKind)),
                memcpy->start, memcpy->end, memcpy->deviceId, memcpy->streamId,
                memcpy->correlationId, memcpy->bytes);
            break;
          }
          case CUPTI_ACTIVITY_KIND_MEMCPY2: {
            auto *memcpy =
                reinterpret_cast<const CUpti_ActivityMemcpy2 *>(record);
            tracer->AddMemRecords(
                MemcpyKind(
                    static_cast<CUpti_ActivityMemcpyKind>(memcpy->copyKind)),
                memcpy->start, memcpy->end, memcpy->deviceId, memcpy->streamId,
                memcpy->correlationId, memcpy->bytes);
            break;
          }
          case CUPTI_ACTIVITY_KIND_DRIVER: {
            auto *api = reinterpret_cast<const CUpti_ActivityAPI *>(record);
            if (api->start != 0 && api->end != 0)
              // -1 device id represents CUDA api call
              tracer->AddCPURecords(
                  DriverKind(api->cbid), api->start, api->end, -1,
                  GetThreadIdFromSystemThreadId(api->threadId));
            break;
          }
          case CUPTI_ACTIVITY_KIND_RUNTIME: {
            auto *api = reinterpret_cast<const CUpti_ActivityAPI *>(record);
            if (api->start != 0 && api->end != 0)
              tracer->AddCPURecords(
                  RuntimeKind(api->cbid), api->start, api->end, -1,
                  GetThreadIdFromSystemThreadId(api->threadId));
            break;
          }
          default: { break; }
        }
      } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
        // Seems not an error in this case.
        break;
      } else {
        CUPTI_CALL(status);
      }
    } while (1);

    size_t dropped;
    CUPTI_CALL(
        dynload::cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
      fprintf(stderr, "Dropped %u activity records\n", (unsigned int)dropped);
    }
  }
  free(buffer);
}
}  // namespace

#endif  // PADDLE_WITH_CUPTI

class DeviceTracerImpl : public DeviceTracer {
 public:
  DeviceTracerImpl() : enabled_(false) {}

  void AddAnnotation(uint32_t id, Event *event) {
    thread_local std::forward_list<std::pair<uint32_t, Event *>>
        *local_correlations_pairs = nullptr;
    if (local_correlations_pairs == nullptr) {
      std::lock_guard<std::mutex> l(trace_mu_);
      correlations_pairs.emplace_front();
      local_correlations_pairs = &correlations_pairs.front();
    }
    local_correlations_pairs->push_front(std::make_pair(id, event));
  }

  void AddCPURecords(const std::string &anno, uint64_t start_ns,
                     uint64_t end_ns, int64_t device_id, int64_t thread_id) {
    if (anno.empty()) {
      VLOG(1) << "Empty timeline annotation.";
      return;
    }
    thread_local std::forward_list<CPURecord> *local_cpu_records_ = nullptr;
    if (local_cpu_records_ == nullptr) {
      std::lock_guard<std::mutex> l(trace_mu_);
      cpu_records_.emplace_front();
      local_cpu_records_ = &cpu_records_.front();
    }
    local_cpu_records_->push_front(
        CPURecord{anno, start_ns, end_ns, device_id, thread_id});
  }

  void AddMemRecords(const std::string &name, uint64_t start_ns,
                     uint64_t end_ns, int64_t device_id, int64_t stream_id,
                     uint32_t correlation_id, uint64_t bytes) {
    // 0 means timestamp information could not be collected for the kernel.
    if (start_ns == 0 || end_ns == 0) {
      VLOG(3) << name << " cannot be traced";
      return;
    }
    // NOTE(liangdun): lock is not needed, only one thread call this function.
    // std::lock_guard<std::mutex> l(trace_mu_);
    mem_records_.push_front(MemRecord{name, start_ns, end_ns, device_id,
                                      stream_id, correlation_id, bytes});
  }

  void AddKernelRecords(std::string name, uint64_t start, uint64_t end,
                        int64_t device_id, int64_t stream_id,
                        uint32_t correlation_id) {
    // 0 means timestamp information could not be collected for the kernel.
    if (start == 0 || end == 0) {
      VLOG(3) << correlation_id << " cannot be traced";
      return;
    }
    // NOTE(liangdun): lock is not needed, only one thread call this function.
    // std::lock_guard<std::mutex> l(trace_mu_);
    kernel_records_.push_front(
        KernelRecord{name, start, end, device_id, stream_id, correlation_id});
  }

  bool IsEnabled() {
    std::lock_guard<std::mutex> l(trace_mu_);
    return enabled_;
  }

  void Enable() {
    std::lock_guard<std::mutex> l(trace_mu_);
    if (enabled_) {
      return;
    }

#ifdef PADDLE_WITH_CUPTI
    EnableActivity();

    // Register callbacks for buffer requests and completed by CUPTI.
    CUPTI_CALL(dynload::cuptiActivityRegisterCallbacks(bufferRequested,
                                                       bufferCompleted));

    CUptiResult ret;
    ret = dynload::cuptiSubscribe(
        &subscriber_, static_cast<CUpti_CallbackFunc>(ApiCallback), this);
    if (ret == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      fprintf(stderr, "CUPTI subcriber limit reached.\n");
    } else if (ret != CUPTI_SUCCESS) {
      fprintf(stderr, "Failed to create CUPTI subscriber.\n");
    }
    const std::vector<int> cbids{
        CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020,
        CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000,
        CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000,
        CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernelMultiDevice_v9000};
    for (auto cbid : cbids)
      CUPTI_CALL(dynload::cuptiEnableCallback(
          1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API, cbid));
    CUPTI_CALL(dynload::cuptiGetTimestamp(&start_ns_));
#endif  // PADDLE_WITH_CUPTI
    enabled_ = true;
  }

  void Reset() {
#ifdef PADDLE_WITH_CUPTI
    CUPTI_CALL(
        dynload::cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
#endif
    std::lock_guard<std::mutex> l(trace_mu_);
    kernel_records_.clear();
    mem_records_.clear();
    correlations_.clear();
    for (auto &tmp : correlations_pairs) tmp.clear();
    for (auto &tmp : cpu_records_) tmp.clear();
  }

  void GenEventKernelCudaElapsedTime() {
#ifdef PADDLE_WITH_CUPTI
    if (correlations_.empty())
      for (auto &tmp : correlations_pairs)
        for (auto &pair : tmp) correlations_[pair.first] = pair.second;
    for (const KernelRecord &r : kernel_records_) {
      if (correlations_.find(r.correlation_id) != correlations_.end()) {
        Event *e = correlations_.at(r.correlation_id);
        e->AddCudaElapsedTime(r.start_ns, r.end_ns);
      }
    }
    for (const auto &r : mem_records_) {
      auto c = correlations_.find(r.correlation_id);
      if (c != correlations_.end() && c->second != nullptr) {
        Event *e = c->second;
        e->AddCudaElapsedTime(r.start_ns, r.end_ns);
      }
    }
#endif
  }

  proto::Profile GenProfile(const std::string &profile_path) {
    int miss = 0, find = 0;
    std::lock_guard<std::mutex> l(trace_mu_);
    proto::Profile profile_pb;
    profile_pb.set_start_ns(start_ns_);
    profile_pb.set_end_ns(end_ns_);
    if (correlations_.empty())
      for (auto &tmp : correlations_pairs)
        for (auto &pair : tmp) correlations_[pair.first] = pair.second;
    for (const KernelRecord &r : kernel_records_) {
      auto *event = profile_pb.add_events();
      event->set_type(proto::Event::GPUKernel);
      auto c = correlations_.find(r.correlation_id);
      if (c != correlations_.end()) {
        event->set_name(c->second->name());
        event->set_detail_info(r.name);
        find++;
      } else {
        VLOG(100) << "Missing Kernel Event: " + r.name;
        miss++;
        event->set_name(r.name);
      }
      event->set_start_ns(r.start_ns);
      event->set_end_ns(r.end_ns);
      event->set_sub_device_id(r.stream_id);
      event->set_device_id(r.device_id);
    }
    VLOG(1) << "KernelRecord event miss: " << miss << " find: " << find;
    for (auto &tmp : cpu_records_)
      for (const CPURecord &r : tmp) {
        auto *event = profile_pb.add_events();
        event->set_type(proto::Event::CPU);
        event->set_name(r.name);
        event->set_start_ns(r.start_ns);
        event->set_end_ns(r.end_ns);
        event->set_sub_device_id(r.thread_id);
        event->set_device_id(r.device_id);
      }
    miss = find = 0;
    for (const MemRecord &r : mem_records_) {
      auto *event = profile_pb.add_events();
      event->set_type(proto::Event::GPUKernel);
      auto c = correlations_.find(r.correlation_id);
      if (c != correlations_.end() && c->second != nullptr) {
        event->set_name(c->second->name());
        event->set_detail_info(r.name);
        find++;
      } else {
        miss++;
        event->set_name(r.name);
      }
      event->set_name(r.name);
      event->set_start_ns(r.start_ns);
      event->set_end_ns(r.end_ns);
      event->set_sub_device_id(r.stream_id);
      event->set_device_id(r.device_id);
      event->mutable_memcopy()->set_bytes(r.bytes);
    }
    VLOG(1) << "MemRecord event miss: " << miss << " find: " << find;
    std::ofstream profile_f;
    profile_f.open(profile_path, std::ios::out | std::ios::trunc);
    std::string profile_str;
    profile_pb.SerializeToString(&profile_str);
    profile_f << profile_str;
    profile_f.close();
    return profile_pb;
  }

  void Disable() {
#ifdef PADDLE_WITH_CUPTI
    // flush might cause additional calls to DeviceTracker.
    CUPTI_CALL(
        dynload::cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
#endif  // PADDLE_WITH_CUPTI
    std::lock_guard<std::mutex> l(trace_mu_);
#ifdef PADDLE_WITH_CUPTI
    DisableActivity();
    CUPTI_CALL(dynload::cuptiUnsubscribe(subscriber_));
    CUPTI_CALL(dynload::cuptiGetTimestamp(&end_ns_));
#endif  // PADDLE_WITH_CUPTI
    enabled_ = false;
  }

 private:
#ifdef PADDLE_WITH_CUPTI
  static void CUPTIAPI ApiCallback(void *userdata, CUpti_CallbackDomain domain,
                                   CUpti_CallbackId cbid, const void *cbdata) {
    auto *cbInfo = reinterpret_cast<const CUpti_CallbackData *>(cbdata);
    DeviceTracerImpl *tracer = reinterpret_cast<DeviceTracerImpl *>(userdata);
    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
      Event *event = CurAnnotation();
      tracer->AddAnnotation(cbInfo->correlationId, event);
    }
  }
  CUpti_SubscriberHandle subscriber_;
#endif  // PADDLE_WITH_CUPTI
  std::mutex trace_mu_;
  bool enabled_;
  uint64_t start_ns_;
  uint64_t end_ns_;
  std::forward_list<KernelRecord> kernel_records_;
  std::forward_list<MemRecord> mem_records_;
  std::forward_list<std::forward_list<CPURecord>> cpu_records_;
  std::forward_list<std::forward_list<std::pair<uint32_t, Event *>>>
      correlations_pairs;
  std::unordered_map<uint32_t, Event *> correlations_;
};

void CreateTracer(DeviceTracer **t) { *t = new DeviceTracerImpl(); }

DeviceTracer *GetDeviceTracer() {
  std::call_once(tracer_once_flag, CreateTracer, &tracer);
  return tracer;
}

void SetCurAnnotation(Event *event) { annotation_stack.push_back(event); }

void ClearCurAnnotation() { annotation_stack.pop_back(); }

Event *CurAnnotation() {
  if (annotation_stack.empty()) return nullptr;
  return annotation_stack.back();
}
std::string CurAnnotationName() {
  if (annotation_stack.empty()) return "";
  return annotation_stack.back()->name();
}

void SetCurBlock(int block_id) { block_id_stack.push_back(block_id); }

void ClearCurBlock() { block_id_stack.pop_back(); }

int BlockDepth() { return block_id_stack.size(); }

uint32_t GetCurSystemThreadId() {
  std::stringstream ss;
  ss << std::this_thread::get_id();
  uint32_t id = static_cast<uint32_t>(std::stoull(ss.str()));
  return id;
}

void RecoreCurThreadId(int32_t id) {
  auto gid = GetCurSystemThreadId();
  VLOG(1) << "RecoreCurThreadId: " << gid << " -> " << id;
  system_thread_id_map[gid] = id;
}

int32_t GetThreadIdFromSystemThreadId(uint32_t id) {
  auto it = system_thread_id_map.find(id);
  if (it != system_thread_id_map.end()) return it->second;
  // return origin id if no event is recorded in this thread.
  return static_cast<int32_t>(id);
}

}  // namespace platform
}  // namespace paddle
