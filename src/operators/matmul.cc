#include "operators/matmul.h"
#include "utils/operator_utils.h"
#include <algorithm>
#include <cassert>

namespace infini {

MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                     bool transB)
    : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}), transA(transA),
      transB(transB) {
  IT_ASSERT(checkValid(graph));
}

string MatmulObj::toString() const {
  std::ostringstream os;
  os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
     << ",A=" << inputs[0]->getGuid() << ",B=" << inputs[1]->getGuid()
     << ",C=" << outputs[0]->getGuid() << ",mnk=[" << m << "," << n << "," << k
     << "])";
  return os.str();
}

optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs) {
  // TODO：返回经过 matmul 操作后的 shape
  // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
  auto A = inputs[0]->getDims();
  auto B = inputs[1]->getDims();
  std::reverse(A.begin(), A.end()), std::reverse(B.begin(), B.end());

  if (transA)
    std::swap(A[0], A[1]);
  if (transB)
    std::swap(B[0], B[1]);

  while (A.size() < B.size())
    A.push_back(1);
  while (B.size() < A.size())
    B.push_back(1);

  for (int i = 2, rA = A.size(); i < rA; i++)
    A[i] = std::max(A[i], B[i]); // broadcast
  A[0] = B[0];
  std::reverse(A.begin(), A.end());
  return {{A}};
}

} // namespace infini