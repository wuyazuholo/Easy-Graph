#include "pagerank.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <utility>

#include "../../../common/pagerank_parallel.h"

namespace egdist {
PageRankResult pagerank(const DistributedGraph& graph, double alpha, int max_iter, double tol, bool weighted) {
    const auto& g = graph;
    const auto& comm = *g.comm;
    comm.check_error(!std::isfinite(alpha) || alpha < 0 || alpha >= 1 ? "alpha must be in [0, 1)" :
                     max_iter <= 0 ? "max_iter must be positive" :
                     !std::isfinite(tol) || tol <= 0 ? "tol must be finite and positive" : "");
    std::ostringstream params;
    params << std::setprecision(17) << alpha << ',' << max_iter << ',' << tol << ',' << weighted;
    comm.agree(params.str());
    const auto n = g.global_num_vertices;
    if (!n) return {g.vertices, {}, 0};
    std::vector<double> old(g.vertices.size(), 1.0 / n), next(old.size()), normalizer(old.size());
    easygraph_parallel::for_each(old.size(), [&](std::size_t i) {
        if (!weighted) normalizer[i] = g.offsets[i + 1] - g.offsets[i];
        else for (auto e = g.offsets[i]; e < g.offsets[i + 1]; ++e) normalizer[i] += g.weights[e];
    });
    bool invalid_sum = false;
    for (auto sum : normalizer) invalid_sum = invalid_sum || !std::isfinite(sum);
    comm.check_error(invalid_sum ? "Outgoing weight sum overflowed" : "");
    std::vector<double> contributions(g.contribution_slots);
    std::vector<double> received(g.value_exchange->recv_size());
    std::vector<MPI_Request> requests(static_cast<std::size_t>(comm.size()) * 2);
    std::vector<double> pull_source_values;
    std::vector<double> pull_received;
    std::vector<MPI_Request> pull_requests;
    if (g.use_dense_pull) {
        pull_source_values.resize(g.vertices.size() + g.pull_value_exchange->recv_size());
        pull_received.resize(g.pull_value_exchange->send_size());
        pull_requests.resize(static_cast<std::size_t>(comm.size()) * 2);
    }
    for (int iteration = 1; iteration <= max_iter; ++iteration) {
        if (g.use_dense_pull) {
            const double local_dangling = easygraph_parallel::sum(old.size(), [&](std::size_t i) {
                return normalizer[i] == 0 ? old[i] : 0.0;
            });
            easygraph_parallel::for_each(old.size(), [&](std::size_t i) {
                contributions[i] = 0.0;
                pull_source_values[i] = normalizer[i] == 0 ? 0.0 : alpha * old[i] / normalizer[i];
            });
            easygraph_parallel::for_each(pull_received.size(), [&](std::size_t i) {
                pull_received[i] = pull_source_values[g.pull_send_source_slots[i]];
            });
            const double* send = pull_received.empty() ? nullptr : pull_received.data();
            double* recv = g.pull_value_exchange->recv_size() ?
                           pull_source_values.data() + old.size() : nullptr;
            g.pull_value_exchange->exchange(comm, send, recv, pull_requests);
            easygraph_parallel::for_each(old.size(), [&](std::size_t v) {
                double value = 0.0;
                for (auto e = g.incoming_offsets[v]; e < g.incoming_offsets[v + 1]; ++e) {
                    const double edge_weight = weighted ? g.incoming_weights[e] : 1.0;
                    value += pull_source_values[g.incoming_source_slots[e]] * edge_weight;
                }
                contributions[v] = value;
            });
            const double jump = (1.0 - alpha + alpha * comm.sum(local_dangling)) / n;
            const double local_error = easygraph_parallel::sum(old.size(), [&](std::size_t i) {
                next[i] = contributions[i] + jump;
                return std::abs(next[i] - old[i]);
            });
            old.swap(next);
            if (comm.sum(local_error) < tol * n) return {g.vertices, std::move(old), iteration};
            continue;
        }
        easygraph_parallel::for_each(contributions.size(), [&](std::size_t i) {
            contributions[i] = 0.0;
        });
        const double local_dangling = easygraph_parallel::sum(old.size(), [&](std::size_t i) {
            return normalizer[i] == 0 ? old[i] : 0.0;
        });
        easygraph_parallel::for_each(old.size(), [&](std::size_t i) {
            if (normalizer[i] == 0) return;
            const double scale = weighted ? alpha * old[i] : alpha * old[i] / normalizer[i];
            for (auto e = g.offsets[i]; e < g.offsets[i + 1]; ++e) {
                const double value = weighted ? scale * (g.weights[e] / normalizer[i]) : scale;
                easygraph_parallel::add(contributions[g.target_slots[e]], value);
            }
        });
        const double* outgoing = g.value_exchange->send_size() ?
            contributions.data() + old.size() : nullptr;
        g.value_exchange->exchange(comm, outgoing, received.data(), requests);
        easygraph_parallel::for_each(received.size(), [&](std::size_t i) {
            easygraph_parallel::add(contributions[g.receive_slots[i]], received[i]);
        });
        const double jump = (1.0 - alpha + alpha * comm.sum(local_dangling)) / n;
        const double local_error = easygraph_parallel::sum(old.size(), [&](std::size_t i) {
            next[i] = contributions[i] + jump;
            return std::abs(next[i] - old[i]);
        });
        old.swap(next);
        if (comm.sum(local_error) < tol * n) return {g.vertices, std::move(old), iteration};
    }
    throw std::runtime_error("Distributed PageRank did not converge within max_iter");
}
}
