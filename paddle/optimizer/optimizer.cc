#include "optimizer.h"
#include <string>

#include "parameter_optimizer.h"

using namespace paddle;
using namespace paddle::optimizer;

template <paddle_element_type VALUE>
struct EnumToType {};

template <class T>
struct TypeToEnum {};

#define MATCH_ENUM_TYPE(TYPE, ENUM)                  \
  template <>                                        \
  struct TypeToEnum<TYPE> {                          \
    static paddle_element_type v() { return ENUM; }; \
    static constexpr TYPE value = ENUM;              \
  };                                                 \
  template <>                                        \
  struct EnumToType<ENUM> {                          \
    typedef TYPE Type;                               \
  }

MATCH_ENUM_TYPE(int32_t, PADDLE_ELEMENT_TYPE_INT32);
MATCH_ENUM_TYPE(uint32_t, PADDLE_ELEMENT_TYPE_UINT32);
MATCH_ENUM_TYPE(int64_t, PADDLE_ELEMENT_TYPE_INT64);
MATCH_ENUM_TYPE(uint64_t, PADDLE_ELEMENT_TYPE_UINT64);
// TODO(zhihong): only implement below type, need to fix
MATCH_ENUM_TYPE(float, PADDLE_ELEMENT_TYPE_FLOAT32);
MATCH_ENUM_TYPE(double, PADDLE_ELEMENT_TYPE_FLOAT64);

struct paddle_optimizer {
  paddle::optimizer::ParameterOptimizer* impl;
};

paddle_optimizer* paddle_create_optimizer(const unsigned char* config_proto,
                                          const int config_proto_len,
                                          const paddle_element_type data_type,
                                          void* param_buffer,
                                          int num_bytes,
                                          const char* state,
                                          const int state_len) {
  paddle_optimizer* optimizer = new paddle_optimizer;
  std::string config(config_proto, config_proto + config_proto_len);
  Tensor* parameter = new Tensor(reinterpret_cast<float*>(param_buffer),
                                 num_bytes / sizeof(float));
  optimizer->impl = ParameterOptimizer::Create(config, parameter);
  if (state != nullptr) {
    std::string s(state, state + state_len);
    optimizer->impl->DeserializeState(s);
  }
  return optimizer;
}

int paddle_release_optimizer(paddle_optimizer* o) {
  if (o != nullptr) delete o->impl;
  return PADDLE_SUCCESS;
}

int paddle_update_parameter(paddle_optimizer* o,
                            const paddle_element_type data_type,
                            const void* grad_buffer,
                            int num_bytes) {
  // TOOD(zhihong): datatype not work. need to add the runtime datatype
  auto grad_type = reinterpret_cast<const float*>(grad_buffer);
  Tensor* gradient =
      new Tensor(const_cast<float*>(grad_type), num_bytes / sizeof(float));
  o->impl->Update(gradient);
  return PADDLE_SUCCESS;
}

int paddle_optimizer_get_weights(paddle_optimizer* o, void** param_buffer) {
  int param_size = 0;
  *param_buffer = (void*)o->impl->get_weight(&param_size);
  return param_size;
}

int paddle_optimizer_get_state(paddle_optimizer* o, const char** state) {
  int state_len = 0;
  *state = o->impl->SerializeState(&state_len);
  return state_len;
}
