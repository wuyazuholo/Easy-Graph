#pragma once

#include "communicator.h"

namespace egdist {
// Immutable routing for repeated exchanges of a fixed, rank-ordered layout.
// The caller owns the reusable value buffers, so separate PageRank calls do
// not share mutable working storage.
class ValueExchangePlan {
public:
    ValueExchangePlan(const Communicator& comm,
                      const std::vector<std::size_t>& send_sizes,
                      std::size_t chunk_elements = INT_MAX)
        : send_starts_(comm.size()), recv_starts_(comm.size()),
          send_sizes_(comm.size()), recv_sizes_(comm.size()) {
        comm.check_error(send_sizes.size() != send_sizes_.size() ||
                         chunk_elements == 0 || chunk_elements > INT_MAX ||
                         comm.size() > INT_MAX / 2 ?
                         "Invalid fixed exchange layout or chunk limit" : "");
        comm.agree(std::to_string(chunk_elements));
        chunk_elements_ = chunk_elements;
        for (std::size_t i = 0; i < send_sizes.size(); ++i)
            send_sizes_[i] = send_sizes[i];
        mpi_check(MPI_Alltoall(send_sizes_.data(), 1, MPI_UNSIGNED_LONG_LONG,
                              recv_sizes_.data(), 1, MPI_UNSIGNED_LONG_LONG, comm.handle()));
        bool overflow = false;
        const auto limit = std::numeric_limits<std::size_t>::max() / sizeof(double);
        unsigned long long local_rounds = 0;
        for (std::size_t i = 0; i < send_sizes_.size(); ++i) {
            send_starts_[i] = send_size_;
            recv_starts_[i] = recv_size_;
            if (send_sizes_[i] > limit - send_size_ || recv_sizes_[i] > limit - recv_size_)
                overflow = true;
            else {
                send_size_ += static_cast<std::size_t>(send_sizes_[i]);
                recv_size_ += static_cast<std::size_t>(recv_sizes_[i]);
            }
            const auto count = std::max(send_sizes_[i], recv_sizes_[i]);
            local_rounds = std::max(local_rounds,
                count / chunk_elements_ + static_cast<unsigned long long>(count % chunk_elements_ != 0));
        }
        comm.check_error(overflow ? "Fixed exchange size overflow" : "");
        mpi_check(MPI_Allreduce(&local_rounds, &rounds_, 1, MPI_UNSIGNED_LONG_LONG,
                               MPI_MAX, comm.handle()));
        int direct = send_size_ <= static_cast<std::size_t>(INT_MAX) &&
                     recv_size_ <= static_cast<std::size_t>(INT_MAX) && rounds_ <= 1;
        int all_direct = 0;
        mpi_check(MPI_Allreduce(&direct, &all_direct, 1, MPI_INT, MPI_MIN, comm.handle()));
        direct_ = all_direct != 0;
        if (direct_) {
            for (std::size_t i = 0; i < send_sizes_.size(); ++i) {
                send_counts_.push_back(static_cast<int>(send_sizes_[i]));
                recv_counts_.push_back(static_cast<int>(recv_sizes_[i]));
                send_offsets_.push_back(static_cast<int>(send_starts_[i]));
                recv_offsets_.push_back(static_cast<int>(recv_starts_[i]));
            }
        }
    }

    std::size_t send_size() const { return send_size_; }
    std::size_t recv_size() const { return recv_size_; }

    // requests is allocated once by the worker and reused across iterations.
    void exchange(const Communicator& comm, const double* send, double* received,
                  std::vector<MPI_Request>& requests) const {
        if (!rounds_) return;
        if (direct_) {
            mpi_check(MPI_Alltoallv(send, send_counts_.data(), send_offsets_.data(), MPI_DOUBLE,
                                   received, recv_counts_.data(), recv_offsets_.data(), MPI_DOUBLE,
                                   comm.handle()));
            return;
        }
        // Large layouts use size_t buffer offsets and bounded per-peer messages;
        // no packing buffer or int displacement is needed on this path either.
        for (unsigned long long round = 0; round < rounds_; ++round) {
            const auto start = static_cast<std::size_t>(round * chunk_elements_);
            int active = 0;
            for (int peer = 0; peer < comm.size(); ++peer) {
                if (start < recv_sizes_[peer]) {
                    const int count = static_cast<int>(std::min<unsigned long long>(
                        chunk_elements_, recv_sizes_[peer] - start));
                    mpi_check(MPI_Irecv(received + recv_starts_[peer] + start, count, MPI_DOUBLE,
                                        peer, 0, comm.handle(), &requests[active++]));
                }
            }
            for (int peer = 0; peer < comm.size(); ++peer) {
                if (start < send_sizes_[peer]) {
                    const int count = static_cast<int>(std::min<unsigned long long>(
                        chunk_elements_, send_sizes_[peer] - start));
                    mpi_check(MPI_Isend(send + send_starts_[peer] + start, count, MPI_DOUBLE,
                                        peer, 0, comm.handle(), &requests[active++]));
                }
            }
            mpi_check(MPI_Waitall(active, requests.data(), MPI_STATUSES_IGNORE));
        }
    }

private:
    std::vector<std::size_t> send_starts_, recv_starts_;
    std::vector<unsigned long long> send_sizes_, recv_sizes_;
    std::vector<int> send_counts_, recv_counts_, send_offsets_, recv_offsets_;
    std::size_t send_size_ = 0, recv_size_ = 0, chunk_elements_ = 0;
    unsigned long long rounds_ = 0;
    bool direct_ = false;
};
}
