#include "distributed_graph.h"

#include <algorithm>
#include <utility>

namespace egdist {
DistributedGraph::DistributedGraph(std::shared_ptr<Communicator> comm_, bool directed_,
                                  std::vector<VertexId> owned, std::vector<EdgeRecord> edges,
                                  std::shared_ptr<const Partitioner> partitioner_)
    : comm(std::move(comm_)),
      partitioner(partitioner_ ? std::move(partitioner_) : std::make_shared<Partitioner>(comm->size())),
      directed(directed_) {
    std::sort(owned.begin(), owned.end());
    owned.erase(std::unique(owned.begin(), owned.end()), owned.end());
    vertices = std::move(owned);
    for (std::size_t i = 0; i < vertices.size(); ++i) local_index.emplace(vertices[i], i);
    std::sort(edges.begin(), edges.end(), [](const EdgeRecord& a, const EdgeRecord& b) {
        if (a.source != b.source) return a.source < b.source;
        if (a.target != b.target) return a.target < b.target;
        return a.order > b.order;
    });
    edges.erase(std::unique(edges.begin(), edges.end(), [](const EdgeRecord& a, const EdgeRecord& b) {
        return a.source == b.source && a.target == b.target;
    }), edges.end());
    offsets.assign(vertices.size() + 1, 0);
    Offset logical_edges = 0;
    for (const auto& edge : edges) {
        ++offsets[local_index.at(edge.source) + 1];
        targets.push_back(edge.target);
        weights.push_back(edge.weight);
        if (directed || edge.source <= edge.target) ++logical_edges;
    }
    for (std::size_t i = 1; i < offsets.size(); ++i) offsets[i] += offsets[i - 1];
    const int rank = comm->rank();
    remote_targets.resize(comm->size());
    for (auto target : targets) {
        const int owner = partitioner->owner(target);
        if (owner != rank) remote_targets[owner].push_back(target);
    }
    std::unordered_map<VertexId, std::size_t> remote_slots;
    contribution_slots = vertices.size();
    for (auto& bucket : remote_targets) {
        std::sort(bucket.begin(), bucket.end());
        bucket.erase(std::unique(bucket.begin(), bucket.end()), bucket.end());
        for (auto target : bucket) remote_slots.emplace(target, contribution_slots++);
    }
    target_slots.reserve(targets.size());
    for (auto target : targets) {
        target_slots.push_back(partitioner->owner(target) == rank ?
                               local_index.at(target) : remote_slots.at(target));
    }
    global_num_vertices = comm->sum(static_cast<Offset>(vertices.size()));
    global_num_edges = comm->sum(logical_edges);
    std::vector<std::size_t> send_sizes;
    for (const auto& bucket : remote_targets) send_sizes.push_back(bucket.size());
    value_exchange.reset(new ValueExchangePlan(*comm, send_sizes));
    // IDs cross the network once; subsequent iterations use the same ordering.
    const auto incoming = comm->exchange(remote_targets);
    receive_slots.reserve(incoming.size());
    bool invalid = incoming.size() != value_exchange->recv_size();
    for (auto vertex : incoming) {
        const auto found = local_index.find(vertex);
        if (found == local_index.end()) invalid = true;
        else receive_slots.push_back(found->second);
    }
    comm->check_error(invalid ? "Invalid fixed PageRank receive layout" : "");

    // GraphScope's dense/sparse cutoff: use the pull layout only for multi-rank
    // graphs whose global average degree exceeds 10.
    constexpr double kDensePullAverageDegree = 10.0;
    use_dense_pull = comm->size() > 1 && global_num_vertices != 0 &&
        static_cast<double>(global_num_edges) / static_cast<double>(global_num_vertices) >
            kDensePullAverageDegree;
    if (!use_dense_pull) return;

    // Route incoming edges to destination owners once during loading.
    std::vector<std::vector<EdgeRecord>> incoming_buckets(comm->size());
    std::vector<EdgeRecord> local_incoming;
    local_incoming.reserve(edges.size());
    for (const auto& edge : edges) {
        const int target_owner = partitioner->owner(edge.target);
        if (target_owner == rank) local_incoming.push_back(edge);
        else incoming_buckets[target_owner].push_back(edge);
    }
    auto exchanged_edges = comm->exchange(incoming_buckets);
    local_incoming.insert(local_incoming.end(), exchanged_edges.begin(), exchanged_edges.end());
    std::sort(local_incoming.begin(), local_incoming.end(), [](const EdgeRecord& a, const EdgeRecord& b) {
        if (a.target != b.target) return a.target < b.target;
        if (a.source != b.source) return a.source < b.source;
        return a.order > b.order;
    });
    local_incoming.erase(std::unique(local_incoming.begin(), local_incoming.end(),
        [](const EdgeRecord& a, const EdgeRecord& b) {
            return a.source == b.source && a.target == b.target;
        }), local_incoming.end());

    remote_sources.resize(comm->size());
    for (const auto& edge : local_incoming) {
        const int source_owner = partitioner->owner(edge.source);
        if (source_owner != rank) remote_sources[source_owner].push_back(edge.source);
    }
    for (auto& bucket : remote_sources) {
        std::sort(bucket.begin(), bucket.end());
        bucket.erase(std::unique(bucket.begin(), bucket.end()), bucket.end());
    }
    std::unordered_map<VertexId, std::size_t> source_slots;
    std::size_t source_slot = vertices.size();
    for (const auto& bucket : remote_sources)
        for (auto source : bucket) source_slots.emplace(source, source_slot++);

    incoming_offsets.assign(vertices.size() + 1, 0);
    incoming_weights.reserve(local_incoming.size());
    incoming_source_slots.reserve(local_incoming.size());
    bool invalid_pull_edge = false;
    for (const auto& edge : local_incoming) {
        auto target_it = local_index.find(edge.target);
        if (target_it == local_index.end()) {
            invalid_pull_edge = true;
            continue;
        }
        ++incoming_offsets[target_it->second + 1];
        incoming_weights.push_back(edge.weight);
        const int source_owner = partitioner->owner(edge.source);
        auto source_it = local_index.find(edge.source);
        if (source_owner == rank) {
            if (source_it == local_index.end()) invalid_pull_edge = true;
            else incoming_source_slots.push_back(source_it->second);
        } else {
            auto remote_it = source_slots.find(edge.source);
            if (remote_it == source_slots.end()) invalid_pull_edge = true;
            else incoming_source_slots.push_back(remote_it->second);
        }
    }
    for (std::size_t i = 1; i < incoming_offsets.size(); ++i)
        incoming_offsets[i] += incoming_offsets[i - 1];
    comm->check_error(invalid_pull_edge ? "Invalid pull edge ownership layout" : "");

    // Exchange requested source IDs once. Replies use the resulting fixed order
    // on all later PageRank iterations, so no vertex IDs are sent in the loop.
    std::vector<unsigned long long> request_send_counts(comm->size()), request_recv_counts(comm->size());
    for (std::size_t peer = 0; peer < remote_sources.size(); ++peer)
        request_send_counts[peer] = remote_sources[peer].size();
    mpi_check(MPI_Alltoall(request_send_counts.data(), 1, MPI_UNSIGNED_LONG_LONG,
                           request_recv_counts.data(), 1, MPI_UNSIGNED_LONG_LONG, comm->handle()));
    const auto requested_sources = comm->exchange(remote_sources);
    std::vector<std::size_t> response_sizes(comm->size());
    std::size_t request_offset = 0;
    bool invalid_request = false;
    for (std::size_t peer = 0; peer < comm->size(); ++peer) {
        const auto count = static_cast<std::size_t>(request_recv_counts[peer]);
        response_sizes[peer] = count;
        for (std::size_t j = 0; j < count; ++j) {
            if (request_offset + j >= requested_sources.size()) {
                invalid_request = true;
                continue;
            }
            auto it = local_index.find(requested_sources[request_offset + j]);
            if (it == local_index.end()) invalid_request = true;
            else pull_send_source_slots.push_back(it->second);
        }
        request_offset += count;
    }
    comm->check_error(invalid_request || request_offset != requested_sources.size() ?
                      "Invalid pull source request layout" : "");
    pull_value_exchange.reset(new ValueExchangePlan(*comm, response_sizes));
    comm->check_error(pull_value_exchange->recv_size() != source_slot - vertices.size() ?
                      "Invalid pull source receive layout" : "");
}
}
