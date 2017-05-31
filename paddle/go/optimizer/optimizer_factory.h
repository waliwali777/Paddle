#ifndef PADDLE_OPTIMIZER_FACTORY_H_
#define PADDLE_OPTIMIZER_FACTORY_H_

#include "parameter_optimizer.h"

namespace paddle {
namespace optimizer {


template <class T>
class SGDOptimizer : public ParameterOptimizer<T> {
public:
  /*! \brief call the applySGD for example  */
  SGDOptimizer(const OptimizerConfig &config);
  void set_weight(const Tensor<T> *p);
  T* get_weight() const;
  void update(const Tensor<T> &gradient);
  char* get_config_proto();
  void destroy();
  ~SGDOptimizer() {
    // clear memory by Tensor library
    delete momentums_;
  }
private:
  Tensor<T>* momentums_;
  double learning_rate;
  double momentum;
  double decay;
  bool nesterov;
  double lr_decay_a;
  double lr_decay_b;
};

template <class T>
class AdagradOptimizer : public ParameterOptimizer<T> {
public:
  void update(const Tensor<T> &gradient) {
  }
private:
};

template <class T>
class AdadeltaOptimizer : public ParameterOptimizer<T> {
public:
  /*! \brief call the applySGD for example  */
  void update(const Tensor<T> &gradient) {
  }
private:
  double learning_rate;
  double rho;
  double epsilon;
  double decay;
};

template <class T>
class AdamOptimizer : public ParameterOptimizer<T> {
public:
  /*! \brief call the applySGD for example  */
  void update(const Tensor<T> &gradient) {
  }
private:
  double learning_rate ;
  double beta_1;
  double beta_2;
  double epsilon;
};

// template <class T>
// class MomentumOptimizer : public ParameterOptimizer {
// public:
//   /*! \brief call the applyXX for example  */
//   MomentumOptimizer(const paddle::optimizer_config &config);
//   void update(const Tensor<T> &gradient) {
//     learning_rate = applyExpLearningRate(config_);
//     applyMomentum(
//         parameter, gradient, momentum, learning_rate, mu, weight_decay);
//   }

// private:
//   double momentum;
// };

}  // namespace optimizer
}  // namespace paddle
#endif
