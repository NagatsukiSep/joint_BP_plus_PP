#pragma once

#include <string>
#include <vector>

struct CudaMsg {
    double v[4];
};

struct CudaBPGraph {
    int nvars = 0;
    int mX = 0;
    int mZ = 0;
    int x_edges = 0;
    int z_edges = 0;
    std::vector<int> x_check_offsets;
    std::vector<int> x_check_edges;
    std::vector<int> x_edge_var;
    std::vector<int> x_edge_check;
    std::vector<int> x_edge_pos;
    std::vector<int> x_var_offsets;
    std::vector<int> x_var_edges;
    std::vector<int> z_check_offsets;
    std::vector<int> z_check_edges;
    std::vector<int> z_edge_var;
    std::vector<int> z_edge_check;
    std::vector<int> z_edge_pos;
    std::vector<int> z_var_offsets;
    std::vector<int> z_var_edges;
};

struct CudaBPResult {
    std::vector<int> est;
    int iterations = 0;
    int check_count = 0;
    bool syndrome_match = false;
    double kernel_ms = 0.0;
    double memcpy_ms = 0.0;
    double check_kernel_ms = 0.0;
    double check_memcpy_ms = 0.0;
    double init_kernel_ms = 0.0;
    double host_ms = 0.0;
};

struct CudaBPContext;

#ifdef USE_CUDA
CudaBPContext *cuda_bp_create(const CudaBPGraph &graph, int device_id, std::string *error_out);
void cuda_bp_destroy(CudaBPContext *ctx);
bool cuda_bp_decode(
    CudaBPContext *ctx,
    const std::vector<int> &sx,
    const std::vector<int> &sz,
    const CudaMsg &prior,
    int max_iter,
    int check_warmup,
    int check_interval,
    bool measure_costs,
    bool use_graph,
    bool freeze_syn,
    double damping,
    CudaBPResult &out,
    std::string *error_out
);
#else
inline CudaBPContext *cuda_bp_create(const CudaBPGraph &, int, std::string *error_out) {
    if (error_out) {
        *error_out = "CUDA support not compiled";
    }
    return nullptr;
}
inline void cuda_bp_destroy(CudaBPContext *) {}
inline bool cuda_bp_decode(
    CudaBPContext *,
    const std::vector<int> &,
    const std::vector<int> &,
    const CudaMsg &,
    int,
    int,
    int,
    bool,
    bool,
    bool,
    double,
    CudaBPResult &,
    std::string *error_out
) {
    if (error_out) {
        *error_out = "CUDA support not compiled";
    }
    return false;
}
#endif
