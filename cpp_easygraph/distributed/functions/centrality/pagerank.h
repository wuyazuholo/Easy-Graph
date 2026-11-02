#pragma once

#include <vector>
#include "../../distributed_graph.h"

namespace egdist {
struct PageRankResult {
    std::vector<VertexId> ids;
    std::vector<double> values;
    int iterations;
};

PageRankResult pagerank(const DistributedGraph& graph, double alpha, int max_iter,
                        double tol, bool weighted);
}
