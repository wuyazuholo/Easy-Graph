#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "communicator.h"
#include "partitioner.h"
#include "types.h"
#include "value_exchange.h"

namespace egdist {
class DistributedGraph {
public:
    DistributedGraph(std::shared_ptr<Communicator> comm, bool directed,
                     std::vector<VertexId> owned, std::vector<EdgeRecord> edges,
                     std::shared_ptr<const Partitioner> partitioner = nullptr);
    std::shared_ptr<Communicator> comm;
    std::shared_ptr<const Partitioner> partitioner;
    bool directed;
    Offset global_num_vertices = 0;
    Offset global_num_edges = 0;
    std::vector<VertexId> vertices;
    std::unordered_map<VertexId, std::size_t> local_index;
    std::vector<Offset> offsets;

    std::vector<VertexId> targets;
    std::vector<double> weights;
    std::vector<std::size_t> target_slots;
    std::vector<std::vector<VertexId>> remote_targets;
    std::size_t contribution_slots = 0;
    std::unique_ptr<ValueExchangePlan> value_exchange;
    std::vector<std::size_t> receive_slots;

    // Reverse layout for the dense pull path; built only when average degree is high.
    bool use_dense_pull = false;
    std::vector<Offset> incoming_offsets;
    std::vector<double> incoming_weights;
    std::vector<std::size_t> incoming_source_slots;
    std::vector<std::vector<VertexId>> remote_sources;
    std::vector<std::size_t> pull_send_source_slots;
    std::unique_ptr<ValueExchangePlan> pull_value_exchange;
};
}
