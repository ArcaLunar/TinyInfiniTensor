#include "core/graph.h"
#include "core/op_type.h"
#include "core/runtime.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>
#include <memory>
#include <numeric>
#include <queue>

namespace infini {

void GraphObj::addOperatorAndConnect(const Operator &op) {
  sorted = false;
  ops.push_back(op);
  for (auto &input : op->getInputs()) {
    if (input) {
      input->addTarget(op);
      if (auto pred = input->getSource()) {
        pred->addSuccessors(op);
        op->addPredecessors(pred);
      }
    }
  }
  for (auto &output : op->getOutputs()) {
    if (output) {
      output->setSource(op);
      for (auto &succ : output->getTargets()) {
        succ->addPredecessors(op);
        op->addSuccessors(succ);
      }
    }
  }
}

string GraphObj::toString() const {
  std::ostringstream oss;
  oss << "Graph Tensors:\n";
  for (const auto &tensor : tensors)
    oss << tensor << "\n";

  oss << "Graph operators:\n";
  for (const auto &op : ops) {
    vector<UidBaseType> preds, succs;
    for (auto &o : op->getPredecessors())
      preds.emplace_back(o->getGuid());
    for (auto &o : op->getSuccessors())
      succs.emplace_back(o->getGuid());
    oss << "OP " << op->getGuid();
    oss << ", pred " << vecToString(preds);
    oss << ", succ " << vecToString(succs);
    oss << ", " << op << "\n";
  }
  return oss.str();
}

bool GraphObj::topo_sort() {
  if (this->sorted) {
    return true;
  }
  std::vector<Operator> sorted;
  std::unordered_set<OperatorObj *> flags;
  sorted.reserve(ops.size());
  flags.reserve(ops.size());
  while (sorted.size() < ops.size()) {
    // Any node is move to sorted in this loop.
    auto modified = false;
    for (auto const &op : ops) {
      if (auto const &inputs = op->getInputs();
          flags.find(op.get()) == flags.end() &&
          std::all_of(inputs.begin(), inputs.end(),
                      [&flags](auto const &input) {
                        auto ptr = input->getSource().get();
                        return !ptr || flags.find(ptr) != flags.end();
                      })) {
        modified = true;
        sorted.emplace_back(op);
        flags.insert(op.get());
      }
    }
    if (!modified) {
      return false;
    }
  }
  this->ops = std::move(sorted);
  return this->sorted = true;
}

void GraphObj::optimize() {
  using vi = std::vector<int>;
  auto isInversePermute = [&](const vi &lhs, const vi &rhs) {
    if (lhs.size() != rhs.size())
      return false;
    for (auto i = 0; i < (int)lhs.size(); ++i)
      if (lhs[rhs[i]] != i)
        return false;
    return true;
  };
  auto transposeLast2dim = [&](const vi &perm) {
    int n = perm.size();
    if (n < 2)
      return false;
    for (int i = 0; i < n - 2; i++)
      if (perm[i] != i)
        return false;
    return perm[n - 1] == n - 2 and perm[n - 2] == n - 1;
  };
  auto safeAddPredecessor = [&](const Operator &node, const Operator &pred) {
    auto preds = node->getPredecessors();
    if (pred && std::find(preds.begin(), preds.end(), pred) == preds.end())
      node->addPredecessors(pred);
  };
  auto safeAddSuccessor = [&](const Operator &node, const Operator &succ) {
    auto succs = node->getSuccessors();
    if (succ && std::find(succs.begin(), succs.end(), succ) == succs.end())
      node->addSuccessors(succ);
  };
  auto safeAddTarget = [&](const Tensor &t, const Operator &target) {
    auto targets = t->getTargets();
    if (target &&
        std::find(targets.begin(), targets.end(), target) == targets.end())
      t->addTarget(target);
  };
  auto removeTransposeOp = [&](const Operator &t) {
    if (!t)
      return;
    for (auto &pred : t->getPredecessors()) {
      pred->removeSuccessors(t);
      t->removePredecessors(pred);
    }
    for (auto &succ : t->getSuccessors()) {
      succ->removePredecessors(t);
      t->removeSuccessors(succ);
    }
    for (auto &input : t->getInputs())
      input->removeTarget(t);
    auto outputs = t->getOutputs();
    for (auto &output : outputs) {
      for (auto &target : output->getTargets())
        target->removePredecessors(t);
      output->targets.clear();
      output->source.reset();
      removeTensor(output);
    }
    removeOperator(t);
  };

  bool updated = false;
  do {
    updated = false;

    // STEP 1: remove 2 transposes
    for (auto &op : ops) {
      if (op->getOpType() != OpType::Transpose)
        continue;
      auto successors = op->getSuccessors();
      if (successors.size() != 1)
        continue;
      auto successor = successors.front();
      if (successor->getOpType() != OpType::Transpose)
        continue;

      auto trans1 = std::dynamic_pointer_cast<TransposeObj>(op);
      auto trans2 = std::dynamic_pointer_cast<TransposeObj>(successor);
      if (!isInversePermute(trans1->getPermute(), trans2->getPermute()))
        continue;

      // find replaces, make modifications
      auto inputTensor = op->getInputs(0);
      auto inputSource = inputTensor->getSource();
      auto midTensor = op->getOutput();
      auto outputTensor = trans2->getOutput();
      auto consumerOps = trans2->getSuccessors();

      // inputSource[op] -> inputTensor -> tran1(op) -> midTensor ->
      // trans2(successor) -> outputTensor -> consumerOps
      // === changes to ===
      // inputSource -> inputTensor -> consumerOps
      for (auto &consumerOp : consumerOps) {
        consumerOp->replaceInput(outputTensor, inputTensor);

        consumerOp->removePredecessors(trans2);
        trans2->removeSuccessors(consumerOp);

        if (inputSource) {
          safeAddSuccessor(inputSource, consumerOp);
          safeAddPredecessor(consumerOp, inputSource);
        }
        outputTensor->removeTarget(consumerOp);
        safeAddTarget(inputTensor, consumerOp);
      }

      inputTensor->removeTarget(trans1);
      if (inputSource) {
        inputSource->removeSuccessors(trans1);
        trans1->removePredecessors(inputSource);
      }
      midTensor->removeTarget(trans2);
      midTensor->source.reset();
      trans2->removePredecessors(trans1);
      trans1->removeSuccessors(trans2);
      outputTensor->source.reset();

      removeTransposeOp(trans1);
      removeTransposeOp(trans2);
      sorted = false;
      updated = true;
      break;
    }

    if (updated)
      continue;

    // STEP 2: merge transposes into matmul
    for (auto &op : ops) {
      if (op->getOpType() != OpType::MatMul)
        continue;

      // input1[tensor] -> trans1[op] -> output1[tensor]
      // input2[tensor] -> trans2[op] -> output2[tensor]
      // output1, output2 -> MatmulObj[op](transA, transB) -> outputMatmul
      auto matmul = std::dynamic_pointer_cast<MatmulObj>(op);
      auto inputsNum = std::min(2, matmul->numInputs());
      assert(inputsNum == 2);
      bool fused = false;

      for (int idx = 0; idx < 2; idx++) {
        auto outputTensor = matmul->getInputs(idx);
        auto op = outputTensor->getSource();
        if (!op or op->getOpType() != OpType::Transpose)
          continue;
        auto trans = std::dynamic_pointer_cast<TransposeObj>(op);
        auto permute = trans->getPermute();
        if (!transposeLast2dim(permute))
          continue;
        auto outputOperators = trans->getSuccessors();
        if (outputOperators.size() != 1)
          continue;

        auto input = trans->getInputs(0);
        matmul->replaceInput(outputTensor, input);
        outputTensor->removeTarget(matmul);
        safeAddTarget(input, matmul);
        matmul->removePredecessors(trans);
        trans->removeSuccessors(matmul);
        auto upstream = input->getSource();
        if (upstream) {
          safeAddPredecessor(matmul, upstream);
          safeAddSuccessor(upstream, matmul);
        }

        if (idx == 0)
          matmul->setTransA(!matmul->getTransA());
        else
          matmul->setTransB(!matmul->getTransB());

        removeTransposeOp(trans);

        sorted = false;
        updated = true;
        fused = true;
        break;
      }
      // Find patterns, make mods
      if (fused)
        break;
    }
  } while (updated);
}

Tensor GraphObj::getTensor(int fuid) const {
  for (auto tensor : tensors) {
    if (tensor->getFuid() == fuid) {
      return tensor;
    }
  }
  return nullptr;
}

void GraphObj::shape_infer() {
  for (auto &op : ops) {
    auto ans = op->inferShape();
    IT_ASSERT(ans.has_value());
    auto oldOutputs = op->getOutputs();
    IT_ASSERT(ans.value().size() == oldOutputs.size());
    // replace the old outputshape and size with new one
    for (int i = 0; i < (int)ans.value().size(); ++i) {
      auto newShape = ans.value()[i];
      auto oldShape = oldOutputs[i]->getDims();
      auto fuid = oldOutputs[i]->getFuid();
      if (newShape != oldShape) {
        auto tensor = this->getTensor(fuid);
        tensor->setShape(newShape);
      }
    }
  }
}

void GraphObj::dataMalloc() {
  // topological sorting first
  IT_ASSERT(topo_sort() == true);

  // TODO：利用 allocator 给计算图分配内存
  // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor
  // 绑定内存

  // iterator over tensor to allocate logical memory
  std::vector<size_t> offsets; // cache offsets
  for (const auto &tensor : tensors) {
    size_t size = tensor->getBytes();
    size_t offset = allocator.alloc(size);
    offsets.push_back(offset);
  }

  auto ptr = allocator.getPtr();
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto &tensor = tensors[i];
    size_t offset = offsets[i];
    Blob blob = make_ref<BlobObj>(runtime, (void *)((size_t)ptr + offset));
    tensor->setDataBlob(blob);
  }

  allocator.info();
}

Tensor GraphObj::addTensor(Shape dim, DataType dtype) {
  return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
}

Tensor GraphObj::addTensor(const Tensor &tensor) {
  IT_ASSERT(tensor->getRuntime() == runtime,
            std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                tensor->getRuntime()->toString() + " to " +
                runtime->toString());
  tensors.emplace_back(tensor);
  return tensor;
}

TensorVec GraphObj::addTensor(const TensorVec &tensors) {
  for (auto &t : tensors)
    addTensor(t);
  return tensors;
}

// tensor's "source" and "target" must be in "ops".
// tensor has no "source" and no "target" must not exist.
// "inputs" or "outputs" of operators must be in "tensors"
// "predecessors" and "successors" of an operator of "ops" must be in "ops".
bool GraphObj::checkValid() const {
  for (auto tensor : tensors) {
    IT_ASSERT(
        !(tensor->getTargets().size() == 0 && nullptr == tensor->getSource()));
    for (auto op : tensor->getTargets()) {
      IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
    }
    auto op = tensor->getSource();
    IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
  }
  for (auto op : ops) {
    for (auto tensor : op->getInputs()) {
      IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                tensors.end());
    }
    for (auto tensor : op->getOutputs()) {
      IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                tensors.end());
    }
    for (auto pre : op->getPredecessors()) {
      IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
    }
    for (auto suc : op->getSuccessors()) {
      IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
    }
  }
  std::set<UidBaseType> s;
  // check whether two tensors with the same FUID exist
  for (auto tensor : tensors) {
    int cnt = s.count(tensor->getFuid());
    IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
    s.insert(tensor->getFuid());
  }
  return true;
}

} // namespace infini
