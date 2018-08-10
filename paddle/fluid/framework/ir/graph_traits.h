// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <stack>
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/node.h"

namespace paddle {
namespace framework {
namespace ir {

template <typename IteratorT>
class iterator_range {
  IteratorT begin_, end_;

 public:
  template <typename Container>
  explicit iterator_range(Container &&c) : begin_(c.begin()), end_(c.end()) {}

  iterator_range(const IteratorT &begin, const IteratorT &end)
      : begin_(begin), end_(end) {}

  const IteratorT &begin() const { return begin_; }
  const IteratorT &end() const { return end_; }
};

// DFS iterator on nodes.
struct NodesDFSIterator
    : public std::iterator<std::forward_iterator_tag, Node *> {
  NodesDFSIterator() = default;
  explicit NodesDFSIterator(const std::vector<Node *> &source);
  NodesDFSIterator(NodesDFSIterator &&other) noexcept;
  NodesDFSIterator(const NodesDFSIterator &other);

  Node &operator*();
  NodesDFSIterator &operator++();
  // TODO(Superjomn) current implementation just compare the first
  // element, need to compare the graph and all the elements in the queue and
  // set.
  NodesDFSIterator &operator=(const NodesDFSIterator &other);
  bool operator==(const NodesDFSIterator &other);
  bool operator!=(const NodesDFSIterator &other) { return !(*this == other); }
  Node *operator->();

 private:
  std::stack<Node *> stack_;
  std::unordered_set<Node *> visited_;
};

/*
 * GraphTraits contains some graph traversal algorithms.
 *
 * Usage:
 *
 */
struct GraphTraits {
  static iterator_range<NodesDFSIterator> DFS(const Graph &g) {
    auto start_points = InferenceStartPoints(g);
    NodesDFSIterator x(start_points);
    return iterator_range<NodesDFSIterator>(NodesDFSIterator(start_points),
                                            NodesDFSIterator());
  }

 private:
  static std::vector<Node *> InferenceStartPoints(const Graph &g) {
    std::vector<Node *> result;
    for (auto *node : g.Nodes()) {
      if (node->inputs.empty()) {
        result.push_back(node);
      }
    }
    return result;
  }
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
