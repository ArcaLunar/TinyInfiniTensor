#include "core/allocator.h"
#include <utility>

namespace infini {
  Allocator::Allocator(Runtime runtime) : runtime(runtime) {
    used = 0;
    peak = 0;
    ptr  = nullptr;

    // 'alignment' defaults to sizeof(uint64_t), because it is the length of
    // the longest data type currently supported by the DataType field of
    // the tensor
    alignment = sizeof(uint64_t);
  }

  Allocator::~Allocator() {
    if (this->ptr != nullptr) { runtime->dealloc(this->ptr); }
  }

  size_t Allocator::alloc(size_t size) {
    IT_ASSERT(this->ptr == nullptr);
    // pad the size to the multiple of alignment
    size = this->getAlignedSize(size);

    // TODO: 设计一个算法来分配内存，返回起始地址偏移量
    size_t addr = 0;
    // first, find a free block that fits
    for (auto it = free_blocks.begin(); it != free_blocks.end(); ++it) {
      if (it->second < size) continue;

      // block found, tailor from end
      it->second -= size;
      addr = it->first + it->second;
      if (it->second == 0) free_blocks.erase(it);
      return addr;
    }

    // if no, allocate the memory at the end
    addr = used;
    used += size;
    if (used > peak) peak = used;
    return addr;
  }

  void Allocator::free(size_t addr, size_t size) {
    IT_ASSERT(this->ptr == nullptr);
    size = getAlignedSize(size);

    // TODO: 设计一个算法来回收内存

    // first, insert
    auto it = free_blocks.insert(std::pair{addr, size}).first; // iterator to the newly inserted block

    // second, find if the newly released block can be merged with existing free blocks
    // 1. look to its left
    if (it != free_blocks.begin()) {
      auto next = std::prev(it);
      while (next != free_blocks.begin()) {
        if (next->first + next->second == it->first) {
          // merge
          next->second += it->second;
          free_blocks.erase(it);
        } else break;
        it = next, next--;
      }
    }

    // 2. look to its right
    if (it != free_blocks.end() && std::next(it) != free_blocks.end()) {
      auto next = std::next(it);
      while (next != free_blocks.end()) {
        if (it->first + it->second == next->first) {
          it->second += next->second;
          free_blocks.erase(next);
        } else break;
        it++, next = std::next(it);
      }
    }
  }

  void *Allocator::getPtr() {
    if (this->ptr == nullptr) {
      this->ptr = runtime->alloc(this->peak);
      printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
    }
    return this->ptr;
  }

  size_t Allocator::getAlignedSize(size_t size) { return ((size - 1) / this->alignment + 1) * this->alignment; }

  void Allocator::info() { std::cout << "Used memory: " << this->used << ", peak memory: " << this->peak << std::endl; }
} // namespace infini
