#pragma once

#include <mpi.h>

#include <climits>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "types.h"

namespace egdist {
void mpi_check(int code);
void finalize_mpi();

class Communicator {
public:
    Communicator();
    ~Communicator();
    Communicator(const Communicator&) = delete;
    Communicator& operator=(const Communicator&) = delete;
    void close();
    bool closed() const { return comm_ == MPI_COMM_NULL; }
    void ensure_open() const;
    int rank() const { ensure_open(); return rank_; }
    int size() const { ensure_open(); return size_; }
    MPI_Comm handle() const { ensure_open(); return comm_; }
    void barrier() const;
    void check_error(const std::string& error) const;
    void agree(const std::string& value) const;
    std::string broadcast_string(const std::string& value, int root) const;
    double sum(double value) const;
    Offset sum(Offset value) const;
    std::pair<std::vector<VertexId>, std::vector<double>> gather_result(
        const std::vector<VertexId>& ids, const std::vector<double>& values,
        int root) const;



    template <typename T>
    std::vector<T> exchange(const std::vector<std::vector<T>>& buckets) const {
        static_assert(std::is_trivially_copyable<T>::value, "POD records required");
        ensure_open();
        check_error(buckets.size() == static_cast<std::size_t>(size_) ? "" :
                    "Incorrect number of destination buckets");
        std::vector<int> counts(size_), offsets(size_), recv_counts(size_), recv_offsets(size_);
        std::size_t total = 0;
        bool overflow = false;
        for (int i = 0; i < size_; ++i) {
            const auto bytes = buckets[i].size() * sizeof(T);
            overflow = overflow || bytes > INT_MAX || total > INT_MAX - bytes;
            if (!overflow) {
                offsets[i] = static_cast<int>(total);
                counts[i] = static_cast<int>(bytes);
                total += bytes;
            }
        }
        check_error(overflow ? "MPI exchange exceeds 2 GiB per rank; chunking is not yet supported" : "");
        mpi_check(MPI_Alltoall(counts.data(), 1, MPI_INT, recv_counts.data(), 1,
                              MPI_INT, handle()));
        std::size_t recv_total = 0;
        for (int i = 0; i < size_; ++i) {
            overflow = overflow || recv_total > static_cast<std::size_t>(INT_MAX - recv_counts[i]);
            if (!overflow) {
                recv_offsets[i] = static_cast<int>(recv_total);
                recv_total += recv_counts[i];
            }
        }
        check_error(overflow ? "Received MPI exchange exceeds 2 GiB per rank" : "");
        std::vector<T> send;
        send.reserve(total / sizeof(T));
        for (const auto& bucket : buckets) send.insert(send.end(), bucket.begin(), bucket.end());
        std::vector<T> received(recv_total / sizeof(T));
        mpi_check(MPI_Alltoallv(send.data(), counts.data(), offsets.data(), MPI_BYTE,
                               received.data(), recv_counts.data(), recv_offsets.data(),
                               MPI_BYTE, handle()));
        return received;
    }

private:
    MPI_Comm comm_ = MPI_COMM_NULL;
    int rank_ = 0;
    int size_ = 0;
};
}
