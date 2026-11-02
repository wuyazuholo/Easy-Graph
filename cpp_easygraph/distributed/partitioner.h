#pragma once
#include "types.h"
#include <string>
#include <unordered_map>

namespace egdist {
class Partitioner {
public:
    explicit Partitioner(int partitions, const std::string& mapping_file = "");
    int owner(VertexId vertex) const;
    bool mapped() const { return mapped_; }
private:
    int partitions_;
    bool mapped_ = false;
    std::unordered_map<VertexId, int> owners_;
};
}
