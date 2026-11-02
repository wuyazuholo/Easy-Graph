#include "partitioner.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace egdist {
Partitioner::Partitioner(int partitions, const std::string& mapping_file) : partitions_(partitions) {
    if (partitions <= 0) throw std::invalid_argument("Partition count must be positive");
    if (mapping_file.empty()) return;
    std::ifstream input(mapping_file);
    if (!input) throw std::runtime_error("Cannot open partition map: " + mapping_file);
    std::string line, magic, extra;
    int file_partitions = 0;
    if (!std::getline(input, line))
        throw std::runtime_error("Empty partition map");
    std::istringstream header(line);
    if (!(header >> magic >> file_partitions) || (header >> extra) ||
        magic != "EGPART1" || file_partitions != partitions)
        throw std::runtime_error("Invalid partition map header or partition count");
    VertexId vertex;
    int rank;
    while (std::getline(input, line)) {
        std::istringstream entry(line);
        if (!(entry >> vertex >> rank) || (entry >> extra) ||
            vertex < 0 || rank < 0 || rank >= partitions ||
            !owners_.emplace(vertex, rank).second)
            throw std::runtime_error("Invalid or duplicate partition map entry");
    }
    if (input.bad() || owners_.empty())
        throw std::runtime_error("Invalid or empty partition map");
    mapped_ = true;
}
int Partitioner::owner(VertexId vertex) const {
    if (vertex < 0) throw std::invalid_argument("Vertex IDs must be nonnegative");
    if (mapped_) {
        const auto found = owners_.find(vertex);
        if (found == owners_.end())
            throw std::runtime_error("Vertex missing from partition map: " + std::to_string(vertex));
        return found->second;
    }
    return static_cast<int>(vertex % partitions_);
}
}
