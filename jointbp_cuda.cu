#include "jointbp_cuda.h"

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cmath>
#include <chrono>
#include <iostream>
#include <sstream>

#ifdef USE_CUDA_FP32
using MsgReal = float;
#else
using MsgReal = double;
#endif

struct DeviceMsg {
    MsgReal v0;
    MsgReal v1;
    MsgReal v2;
    MsgReal v3;
};

__host__ __device__ DeviceMsg make_msg(MsgReal a, MsgReal b, MsgReal c, MsgReal d) {
    DeviceMsg m;
    m.v0 = a;
    m.v1 = b;
    m.v2 = c;
    m.v3 = d;
    return m;
}

namespace {

__device__ inline void normalize_msg(DeviceMsg &m) {
    MsgReal sum = m.v0 + m.v1 + m.v2 + m.v3;
    if (sum <= 0.0) {
        m.v0 = 0.25;
        m.v1 = 0.25;
        m.v2 = 0.25;
        m.v3 = 0.25;
        return;
    }
    MsgReal inv = static_cast<MsgReal>(1.0) / sum;
    m.v0 *= inv;
    m.v1 *= inv;
    m.v2 *= inv;
    m.v3 *= inv;
}

__device__ inline DeviceMsg multiply_msg(const DeviceMsg &a, const DeviceMsg &b) {
    return make_msg(a.v0 * b.v0, a.v1 * b.v1, a.v2 * b.v2, a.v3 * b.v3);
}

__device__ inline MsgReal abs_real(MsgReal v) {
#ifdef USE_CUDA_FP32
    return fabsf(v);
#else
    return fabs(v);
#endif
}

__device__ inline DeviceMsg divide_msg(const DeviceMsg &num, const DeviceMsg &den) {
    const MsgReal eps = static_cast<MsgReal>(1e-20);
    return make_msg(num.v0 / (abs_real(den.v0) + eps),
                    num.v1 / (abs_real(den.v1) + eps),
                    num.v2 / (abs_real(den.v2) + eps),
                    num.v3 / (abs_real(den.v3) + eps));
}

__device__ inline int xbit(int state) {
    return (state == 1 || state == 3) ? 1 : 0;
}

__device__ inline int zbit(int state) {
    return (state == 2 || state == 3) ? 1 : 0;
}

__device__ inline DeviceMsg det_msg_xbit(int bit) {
    return bit ? make_msg(0.0, 0.5, 0.0, 0.5) : make_msg(0.5, 0.0, 0.5, 0.0);
}

__device__ inline DeviceMsg det_msg_zbit(int bit) {
    return bit ? make_msg(0.0, 0.0, 0.5, 0.5) : make_msg(0.5, 0.5, 0.0, 0.0);
}

__global__ void init_messages_kernel(int edges, DeviceMsg *v2c, DeviceMsg *c2v, DeviceMsg prior) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= edges) return;
    v2c[idx] = prior;
    c2v[idx] = make_msg(static_cast<MsgReal>(0.25),
                        static_cast<MsgReal>(0.25),
                        static_cast<MsgReal>(0.25),
                        static_cast<MsgReal>(0.25));
}

__global__ void check_update_x_kernel(
    int edges,
    const int *check_offsets,
    const int *check_edges,
    const int *edge_check,
    const int *sx,
    const DeviceMsg *v2c,
    DeviceMsg *c2v
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= edges) return;
    int c = edge_check[e];
    int start = check_offsets[c];
    int end = check_offsets[c + 1];
    MsgReal p_even = static_cast<MsgReal>(1.0);
    MsgReal p_odd = static_cast<MsgReal>(0.0);
    for (int idx = start; idx < end; ++idx) {
        int ej = check_edges[idx];
        if (ej == e) continue;
        DeviceMsg m = v2c[ej];
        MsgReal q0 = m.v0 + m.v2;
        MsgReal q1 = m.v1 + m.v3;
        MsgReal new_even = p_even * q0 + p_odd * q1;
        MsgReal new_odd = p_even * q1 + p_odd * q0;
        p_even = new_even;
        p_odd = new_odd;
    }
    MsgReal val0 = (sx[c] == 0) ? p_even : p_odd;
    MsgReal val1 = (sx[c] == 0) ? p_odd : p_even;
    DeviceMsg out = make_msg(val0, val1, val0, val1);
    normalize_msg(out);
    c2v[e] = out;
}

__global__ void check_update_x_by_check_kernel(
    int checks,
    const int *check_offsets,
    const int *check_edges,
    const int *sx,
    const DeviceMsg *v2c,
    DeviceMsg *c2v
) {
    int c = blockIdx.x;
    if (c >= checks) return;
    int start = check_offsets[c];
    int end = check_offsets[c + 1];
    int deg = end - start;
    int tid = threadIdx.x;
    if (tid >= deg) return;
    extern __shared__ MsgReal shared[];
    MsgReal *q0 = shared;
    MsgReal *q1 = q0 + deg;
    MsgReal *pref_even = q1 + deg;
    MsgReal *pref_odd = pref_even + deg;
    MsgReal *suff_even = pref_odd + deg;
    MsgReal *suff_odd = suff_even + deg;

    int edge_idx = check_edges[start + tid];
    DeviceMsg m = v2c[edge_idx];
    q0[tid] = m.v0 + m.v2;
    q1[tid] = m.v1 + m.v3;
    __syncthreads();

    pref_even[tid] = q0[tid];
    pref_odd[tid] = q1[tid];
    __syncthreads();
    for (int offset = 1; offset < deg; offset <<= 1) {
        if (tid >= offset) {
            MsgReal a_even = pref_even[tid - offset];
            MsgReal a_odd = pref_odd[tid - offset];
            MsgReal b_even = pref_even[tid];
            MsgReal b_odd = pref_odd[tid];
            suff_even[tid] = a_even * b_even + a_odd * b_odd;
            suff_odd[tid] = a_even * b_odd + a_odd * b_even;
        } else {
            suff_even[tid] = pref_even[tid];
            suff_odd[tid] = pref_odd[tid];
        }
        __syncthreads();
        pref_even[tid] = suff_even[tid];
        pref_odd[tid] = suff_odd[tid];
        __syncthreads();
    }

    MsgReal pref_ex_even = (tid == 0) ? static_cast<MsgReal>(1.0) : pref_even[tid - 1];
    MsgReal pref_ex_odd = (tid == 0) ? static_cast<MsgReal>(0.0) : pref_odd[tid - 1];

    suff_even[tid] = q0[tid];
    suff_odd[tid] = q1[tid];
    __syncthreads();
    for (int offset = 1; offset < deg; offset <<= 1) {
        if (tid + offset < deg) {
            MsgReal a_even = suff_even[tid];
            MsgReal a_odd = suff_odd[tid];
            MsgReal b_even = suff_even[tid + offset];
            MsgReal b_odd = suff_odd[tid + offset];
            q0[tid] = a_even * b_even + a_odd * b_odd;
            q1[tid] = a_even * b_odd + a_odd * b_even;
        } else {
            q0[tid] = suff_even[tid];
            q1[tid] = suff_odd[tid];
        }
        __syncthreads();
        suff_even[tid] = q0[tid];
        suff_odd[tid] = q1[tid];
        __syncthreads();
    }

    MsgReal suff_ex_even = (tid == deg - 1) ? static_cast<MsgReal>(1.0) : suff_even[tid + 1];
    MsgReal suff_ex_odd = (tid == deg - 1) ? static_cast<MsgReal>(0.0) : suff_odd[tid + 1];

    MsgReal p_even = pref_ex_even * suff_ex_even + pref_ex_odd * suff_ex_odd;
    MsgReal p_odd = pref_ex_even * suff_ex_odd + pref_ex_odd * suff_ex_even;
    MsgReal val0 = (sx[c] == 0) ? p_even : p_odd;
    MsgReal val1 = (sx[c] == 0) ? p_odd : p_even;
    DeviceMsg out = make_msg(val0, val1, val0, val1);
    normalize_msg(out);
    c2v[edge_idx] = out;
}

__global__ void check_update_z_kernel(
    int edges,
    const int *check_offsets,
    const int *check_edges,
    const int *edge_check,
    const int *sz,
    const DeviceMsg *v2c,
    DeviceMsg *c2v
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= edges) return;
    int c = edge_check[e];
    int start = check_offsets[c];
    int end = check_offsets[c + 1];
    MsgReal p_even = static_cast<MsgReal>(1.0);
    MsgReal p_odd = static_cast<MsgReal>(0.0);
    for (int idx = start; idx < end; ++idx) {
        int ej = check_edges[idx];
        if (ej == e) continue;
        DeviceMsg m = v2c[ej];
        MsgReal q0 = m.v0 + m.v1;
        MsgReal q1 = m.v2 + m.v3;
        MsgReal new_even = p_even * q0 + p_odd * q1;
        MsgReal new_odd = p_even * q1 + p_odd * q0;
        p_even = new_even;
        p_odd = new_odd;
    }
    MsgReal val0 = (sz[c] == 0) ? p_even : p_odd;
    MsgReal val1 = (sz[c] == 0) ? p_odd : p_even;
    DeviceMsg out = make_msg(val0, val0, val1, val1);
    normalize_msg(out);
    c2v[e] = out;
}

__global__ void check_update_z_by_check_kernel(
    int checks,
    const int *check_offsets,
    const int *check_edges,
    const int *sz,
    const DeviceMsg *v2c,
    DeviceMsg *c2v
) {
    int c = blockIdx.x;
    if (c >= checks) return;
    int start = check_offsets[c];
    int end = check_offsets[c + 1];
    int deg = end - start;
    int tid = threadIdx.x;
    if (tid >= deg) return;
    extern __shared__ MsgReal shared[];
    MsgReal *q0 = shared;
    MsgReal *q1 = q0 + deg;
    MsgReal *pref_even = q1 + deg;
    MsgReal *pref_odd = pref_even + deg;
    MsgReal *suff_even = pref_odd + deg;
    MsgReal *suff_odd = suff_even + deg;

    int edge_idx = check_edges[start + tid];
    DeviceMsg m = v2c[edge_idx];
    q0[tid] = m.v0 + m.v1;
    q1[tid] = m.v2 + m.v3;
    __syncthreads();

    pref_even[tid] = q0[tid];
    pref_odd[tid] = q1[tid];
    __syncthreads();
    for (int offset = 1; offset < deg; offset <<= 1) {
        if (tid >= offset) {
            MsgReal a_even = pref_even[tid - offset];
            MsgReal a_odd = pref_odd[tid - offset];
            MsgReal b_even = pref_even[tid];
            MsgReal b_odd = pref_odd[tid];
            suff_even[tid] = a_even * b_even + a_odd * b_odd;
            suff_odd[tid] = a_even * b_odd + a_odd * b_even;
        } else {
            suff_even[tid] = pref_even[tid];
            suff_odd[tid] = pref_odd[tid];
        }
        __syncthreads();
        pref_even[tid] = suff_even[tid];
        pref_odd[tid] = suff_odd[tid];
        __syncthreads();
    }

    MsgReal pref_ex_even = (tid == 0) ? static_cast<MsgReal>(1.0) : pref_even[tid - 1];
    MsgReal pref_ex_odd = (tid == 0) ? static_cast<MsgReal>(0.0) : pref_odd[tid - 1];

    suff_even[tid] = q0[tid];
    suff_odd[tid] = q1[tid];
    __syncthreads();
    for (int offset = 1; offset < deg; offset <<= 1) {
        if (tid + offset < deg) {
            MsgReal a_even = suff_even[tid];
            MsgReal a_odd = suff_odd[tid];
            MsgReal b_even = suff_even[tid + offset];
            MsgReal b_odd = suff_odd[tid + offset];
            q0[tid] = a_even * b_even + a_odd * b_odd;
            q1[tid] = a_even * b_odd + a_odd * b_even;
        } else {
            q0[tid] = suff_even[tid];
            q1[tid] = suff_odd[tid];
        }
        __syncthreads();
        suff_even[tid] = q0[tid];
        suff_odd[tid] = q1[tid];
        __syncthreads();
    }

    MsgReal suff_ex_even = (tid == deg - 1) ? static_cast<MsgReal>(1.0) : suff_even[tid + 1];
    MsgReal suff_ex_odd = (tid == deg - 1) ? static_cast<MsgReal>(0.0) : suff_odd[tid + 1];

    MsgReal p_even = pref_ex_even * suff_ex_even + pref_ex_odd * suff_ex_odd;
    MsgReal p_odd = pref_ex_even * suff_ex_odd + pref_ex_odd * suff_ex_even;
    MsgReal val0 = (sz[c] == 0) ? p_even : p_odd;
    MsgReal val1 = (sz[c] == 0) ? p_odd : p_even;
    DeviceMsg out = make_msg(val0, val0, val1, val1);
    normalize_msg(out);
    c2v[edge_idx] = out;
}

__global__ void variable_update_kernel(
    int nvars,
    const int *x_var_offsets,
    const int *x_var_edges,
    const int *z_var_offsets,
    const int *z_var_edges,
    const DeviceMsg *x_c2v,
    const DeviceMsg *z_c2v,
    DeviceMsg *x_v2c,
    DeviceMsg *z_v2c,
    DeviceMsg prior,
    double damping,
    const int *freeze_x_flag,
    const int *freeze_z_flag,
    int *est,
    double *abs_llr_x,
    double *abs_llr_z
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nvars) return;
    int freeze_x = freeze_x_flag ? *freeze_x_flag : 0;
    int freeze_z = freeze_z_flag ? *freeze_z_flag : 0;
    DeviceMsg total = prior;
    int x_start = x_var_offsets[v];
    int x_end = x_var_offsets[v + 1];
    int z_start = z_var_offsets[v];
    int z_end = z_var_offsets[v + 1];
    for (int idx = x_start; idx < x_end; ++idx) {
        total = multiply_msg(total, x_c2v[x_var_edges[idx]]);
    }
    for (int idx = z_start; idx < z_end; ++idx) {
        total = multiply_msg(total, z_c2v[z_var_edges[idx]]);
    }
    normalize_msg(total);
    int best = 0;
    MsgReal best_val = total.v0;
    if (total.v1 > best_val) {
        best_val = total.v1;
        best = 1;
    }
    if (total.v2 > best_val) {
        best_val = total.v2;
        best = 2;
    }
    if (total.v3 > best_val) {
        best = 3;
    }
    est[v] = best;
    if (abs_llr_x && abs_llr_z) {
        const double eps = 1e-300;
        double px1 = static_cast<double>(total.v1 + total.v3);
        double px0 = static_cast<double>(total.v0 + total.v2);
        double pz1 = static_cast<double>(total.v2 + total.v3);
        double pz0 = static_cast<double>(total.v0 + total.v1);
        double llr_x = log(fmax(px1, eps)) - log(fmax(px0, eps));
        double llr_z = log(fmax(pz1, eps)) - log(fmax(pz0, eps));
        abs_llr_x[v] = fabs(llr_x);
        abs_llr_z[v] = fabs(llr_z);
    }
    if (!freeze_x) {
        const bool use_damping = damping > 0.0;
        const MsgReal keep = use_damping ? static_cast<MsgReal>(1.0 - damping) : static_cast<MsgReal>(1.0);
        const MsgReal damp = use_damping ? static_cast<MsgReal>(damping) : static_cast<MsgReal>(0.0);
        for (int idx = x_start; idx < x_end; ++idx) {
            int e = x_var_edges[idx];
            DeviceMsg out = divide_msg(total, x_c2v[e]);
            normalize_msg(out);
            if (use_damping) {
                DeviceMsg old = x_v2c[e];
                out.v0 = keep * out.v0 + damp * old.v0;
                out.v1 = keep * out.v1 + damp * old.v1;
                out.v2 = keep * out.v2 + damp * old.v2;
                out.v3 = keep * out.v3 + damp * old.v3;
                normalize_msg(out);
            }
            x_v2c[e] = out;
        }
    }
    if (!freeze_z) {
        const bool use_damping = damping > 0.0;
        const MsgReal keep = use_damping ? static_cast<MsgReal>(1.0 - damping) : static_cast<MsgReal>(1.0);
        const MsgReal damp = use_damping ? static_cast<MsgReal>(damping) : static_cast<MsgReal>(0.0);
        for (int idx = z_start; idx < z_end; ++idx) {
            int e = z_var_edges[idx];
            DeviceMsg out = divide_msg(total, z_c2v[e]);
            normalize_msg(out);
            if (use_damping) {
                DeviceMsg old = z_v2c[e];
                out.v0 = keep * out.v0 + damp * old.v0;
                out.v1 = keep * out.v1 + damp * old.v1;
                out.v2 = keep * out.v2 + damp * old.v2;
                out.v3 = keep * out.v3 + damp * old.v3;
                normalize_msg(out);
            }
            z_v2c[e] = out;
        }
    }
}

__global__ void syndrome_kernel(
    int checks,
    const int *check_offsets,
    const int *check_edges,
    const int *edge_var,
    const int *est,
    int use_xbit,
    int *syndrome
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= checks) return;
    int start = check_offsets[c];
    int end = check_offsets[c + 1];
    int parity = 0;
    for (int idx = start; idx < end; ++idx) {
        int v = edge_var[check_edges[idx]];
        int bit = use_xbit ? xbit(est[v]) : zbit(est[v]);
        parity ^= bit;
    }
    syndrome[c] = parity;
}

__global__ void syndrome_compare_kernel(
    int checks,
    const int *check_offsets,
    const int *check_edges,
    const int *edge_var,
    const int *est,
    const int *syndrome,
    int use_xbit,
    int *syn_all
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= checks) return;
    int start = check_offsets[c];
    int end = check_offsets[c + 1];
    int parity = 0;
    for (int idx = start; idx < end; ++idx) {
        int v = edge_var[check_edges[idx]];
        int bit = use_xbit ? xbit(est[v]) : zbit(est[v]);
        parity ^= bit;
    }
    if (parity != syndrome[c]) {
        atomicAnd(syn_all, 0);
    }
}

__global__ void mismatch_count_kernel(int n, const int *a, const int *b, int *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (a[idx] != b[idx]) {
        atomicAdd(out, 1);
    }
}

__global__ void compare_syndrome_kernel(int n, const int *a, const int *b, int *syn_all) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (a[idx] != b[idx]) {
        atomicAnd(syn_all, 0);
    }
}

__global__ void freeze_msgs_kernel(
    int edges,
    const int *edge_var,
    const int *est,
    int use_xbit,
    DeviceMsg *v2c,
    DeviceMsg *c2v
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= edges) return;
    int v = edge_var[e];
    int bit = use_xbit ? xbit(est[v]) : zbit(est[v]);
    DeviceMsg msg = use_xbit ? det_msg_xbit(bit) : det_msg_zbit(bit);
    v2c[e] = msg;
    c2v[e] = msg;
}

}  // namespace

struct CudaBPContext {
    int nvars = 0;
    int mX = 0;
    int mZ = 0;
    int x_edges = 0;
    int z_edges = 0;
    int max_x_deg = 0;
    int max_z_deg = 0;
    cudaStream_t graph_stream = nullptr;
    cudaGraph_t iter_graph = nullptr;
    cudaGraphExec_t iter_exec = nullptr;
    cudaGraph_t check_graph = nullptr;
    cudaGraphExec_t check_exec = nullptr;
    bool graph_ready = false;
    CudaMsg graph_prior{{0.0, 0.0, 0.0, 0.0}};
    double graph_damping = 0.0;
    long long graph_builds = 0;
    long long graph_reuses = 0;
    int *d_x_check_offsets = nullptr;
    int *d_x_check_edges = nullptr;
    int *d_x_edge_var = nullptr;
    int *d_x_edge_check = nullptr;
    int *d_x_var_offsets = nullptr;
    int *d_x_var_edges = nullptr;
    int *d_z_check_offsets = nullptr;
    int *d_z_check_edges = nullptr;
    int *d_z_edge_var = nullptr;
    int *d_z_edge_check = nullptr;
    int *d_z_var_offsets = nullptr;
    int *d_z_var_edges = nullptr;
    int *d_sx = nullptr;
    int *d_sz = nullptr;
    int *d_sx_hat = nullptr;
    int *d_sz_hat = nullptr;
    int *d_est = nullptr;
    int *d_mismatch_x = nullptr;
    int *d_mismatch_z = nullptr;
    int *d_syn_x = nullptr;
    int *d_syn_z = nullptr;
    int *d_freeze_x = nullptr;
    int *d_freeze_z = nullptr;
    DeviceMsg *d_x_v2c = nullptr;
    DeviceMsg *d_x_c2v = nullptr;
    DeviceMsg *d_z_v2c = nullptr;
    DeviceMsg *d_z_c2v = nullptr;
};

namespace {

bool check_cuda(cudaError_t err, std::string *error_out, const char *ctx) {
    if (err == cudaSuccess) return true;
    if (error_out) {
        std::ostringstream oss;
        oss << ctx << ": " << cudaGetErrorString(err);
        *error_out = oss.str();
    }
    return false;
}

bool copy_int_array(const std::vector<int> &src, int **dst, std::string *error_out, const char *ctx) {
    if (src.empty()) {
        *dst = nullptr;
        return true;
    }
    size_t bytes = sizeof(int) * src.size();
    cudaError_t err = cudaMalloc(reinterpret_cast<void **>(dst), bytes);
    if (!check_cuda(err, error_out, ctx)) return false;
    err = cudaMemcpy(*dst, src.data(), bytes, cudaMemcpyHostToDevice);
    return check_cuda(err, error_out, ctx);
}

}  // namespace

CudaBPContext *cuda_bp_create(const CudaBPGraph &graph, int device_id, std::string *error_out) {
    if (!check_cuda(cudaSetDevice(device_id), error_out, "cudaSetDevice")) {
        return nullptr;
    }
    auto *ctx = new CudaBPContext();
    ctx->nvars = graph.nvars;
    ctx->mX = graph.mX;
    ctx->mZ = graph.mZ;
    ctx->x_edges = graph.x_edges;
    ctx->z_edges = graph.z_edges;
    ctx->max_x_deg = 0;
    for (int c = 0; c < graph.mX; ++c) {
        int deg = graph.x_check_offsets[c + 1] - graph.x_check_offsets[c];
        if (deg > ctx->max_x_deg) ctx->max_x_deg = deg;
    }
    ctx->max_z_deg = 0;
    for (int c = 0; c < graph.mZ; ++c) {
        int deg = graph.z_check_offsets[c + 1] - graph.z_check_offsets[c];
        if (deg > ctx->max_z_deg) ctx->max_z_deg = deg;
    }

    if (!copy_int_array(graph.x_check_offsets, &ctx->d_x_check_offsets, error_out, "x_check_offsets")) return nullptr;
    if (!copy_int_array(graph.x_check_edges, &ctx->d_x_check_edges, error_out, "x_check_edges")) return nullptr;
    if (!copy_int_array(graph.x_edge_var, &ctx->d_x_edge_var, error_out, "x_edge_var")) return nullptr;
    if (!copy_int_array(graph.x_edge_check, &ctx->d_x_edge_check, error_out, "x_edge_check")) return nullptr;
    if (!copy_int_array(graph.x_var_offsets, &ctx->d_x_var_offsets, error_out, "x_var_offsets")) return nullptr;
    if (!copy_int_array(graph.x_var_edges, &ctx->d_x_var_edges, error_out, "x_var_edges")) return nullptr;

    if (!copy_int_array(graph.z_check_offsets, &ctx->d_z_check_offsets, error_out, "z_check_offsets")) return nullptr;
    if (!copy_int_array(graph.z_check_edges, &ctx->d_z_check_edges, error_out, "z_check_edges")) return nullptr;
    if (!copy_int_array(graph.z_edge_var, &ctx->d_z_edge_var, error_out, "z_edge_var")) return nullptr;
    if (!copy_int_array(graph.z_edge_check, &ctx->d_z_edge_check, error_out, "z_edge_check")) return nullptr;
    if (!copy_int_array(graph.z_var_offsets, &ctx->d_z_var_offsets, error_out, "z_var_offsets")) return nullptr;
    if (!copy_int_array(graph.z_var_edges, &ctx->d_z_var_edges, error_out, "z_var_edges")) return nullptr;

    if (!check_cuda(cudaMalloc(reinterpret_cast<void **>(&ctx->d_sx), sizeof(int) * graph.mX), error_out, "sx")) return nullptr;
    if (!check_cuda(cudaMalloc(reinterpret_cast<void **>(&ctx->d_sz), sizeof(int) * graph.mZ), error_out, "sz")) return nullptr;
    if (!check_cuda(cudaMalloc(reinterpret_cast<void **>(&ctx->d_sx_hat), sizeof(int) * graph.mX), error_out, "sx_hat")) return nullptr;
    if (!check_cuda(cudaMalloc(reinterpret_cast<void **>(&ctx->d_sz_hat), sizeof(int) * graph.mZ), error_out, "sz_hat")) return nullptr;
    if (!check_cuda(cudaMalloc(reinterpret_cast<void **>(&ctx->d_est), sizeof(int) * graph.nvars), error_out, "est")) return nullptr;
    if (!check_cuda(cudaMalloc(reinterpret_cast<void **>(&ctx->d_mismatch_x), sizeof(int)), error_out, "mismatch_x")) return nullptr;
    if (!check_cuda(cudaMalloc(reinterpret_cast<void **>(&ctx->d_mismatch_z), sizeof(int)), error_out, "mismatch_z")) return nullptr;
    if (!check_cuda(cudaMalloc(reinterpret_cast<void **>(&ctx->d_syn_x), sizeof(int)), error_out, "syn_x")) return nullptr;
    if (!check_cuda(cudaMalloc(reinterpret_cast<void **>(&ctx->d_syn_z), sizeof(int)), error_out, "syn_z")) return nullptr;
    if (!check_cuda(cudaMalloc(reinterpret_cast<void **>(&ctx->d_freeze_x), sizeof(int)), error_out, "freeze_x")) return nullptr;
    if (!check_cuda(cudaMalloc(reinterpret_cast<void **>(&ctx->d_freeze_z), sizeof(int)), error_out, "freeze_z")) return nullptr;

    if (!check_cuda(cudaMalloc(reinterpret_cast<void **>(&ctx->d_x_v2c), sizeof(DeviceMsg) * graph.x_edges), error_out, "x_v2c")) return nullptr;
    if (!check_cuda(cudaMalloc(reinterpret_cast<void **>(&ctx->d_x_c2v), sizeof(DeviceMsg) * graph.x_edges), error_out, "x_c2v")) return nullptr;
    if (!check_cuda(cudaMalloc(reinterpret_cast<void **>(&ctx->d_z_v2c), sizeof(DeviceMsg) * graph.z_edges), error_out, "z_v2c")) return nullptr;
    if (!check_cuda(cudaMalloc(reinterpret_cast<void **>(&ctx->d_z_c2v), sizeof(DeviceMsg) * graph.z_edges), error_out, "z_c2v")) return nullptr;

    return ctx;
}

void cuda_bp_destroy(CudaBPContext *ctx) {
    if (!ctx) return;
    if (ctx->iter_exec) cudaGraphExecDestroy(ctx->iter_exec);
    if (ctx->check_exec) cudaGraphExecDestroy(ctx->check_exec);
    if (ctx->iter_graph) cudaGraphDestroy(ctx->iter_graph);
    if (ctx->check_graph) cudaGraphDestroy(ctx->check_graph);
    if (ctx->graph_stream) cudaStreamDestroy(ctx->graph_stream);
    cudaFree(ctx->d_x_check_offsets);
    cudaFree(ctx->d_x_check_edges);
    cudaFree(ctx->d_x_edge_var);
    cudaFree(ctx->d_x_edge_check);
    cudaFree(ctx->d_x_var_offsets);
    cudaFree(ctx->d_x_var_edges);
    cudaFree(ctx->d_z_check_offsets);
    cudaFree(ctx->d_z_check_edges);
    cudaFree(ctx->d_z_edge_var);
    cudaFree(ctx->d_z_edge_check);
    cudaFree(ctx->d_z_var_offsets);
    cudaFree(ctx->d_z_var_edges);
    cudaFree(ctx->d_sx);
    cudaFree(ctx->d_sz);
    cudaFree(ctx->d_sx_hat);
    cudaFree(ctx->d_sz_hat);
    cudaFree(ctx->d_est);
    cudaFree(ctx->d_mismatch_x);
    cudaFree(ctx->d_mismatch_z);
    cudaFree(ctx->d_syn_x);
    cudaFree(ctx->d_syn_z);
    cudaFree(ctx->d_freeze_x);
    cudaFree(ctx->d_freeze_z);
    cudaFree(ctx->d_x_v2c);
    cudaFree(ctx->d_x_c2v);
    cudaFree(ctx->d_z_v2c);
    cudaFree(ctx->d_z_c2v);
    delete ctx;
}

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
    bool log_graph,
    bool freeze_syn,
    double damping,
    CudaBPResult &out,
    std::string *error_out
) {
    auto host_start = std::chrono::steady_clock::now();
    double memcpy_ms = 0.0;
    double check_memcpy_ms = 0.0;
    auto time_memcpy = [&](cudaError_t err, std::string *err_out, const char *ctx_label, double *bucket) -> bool {
        auto t0 = std::chrono::steady_clock::now();
        bool ok = check_cuda(err, err_out, ctx_label);
        auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        memcpy_ms += dt;
        if (bucket) {
            *bucket += dt;
        }
        return ok;
    };

    if (!ctx) {
        if (error_out) *error_out = "CUDA context not initialized";
        return false;
    }
    if (static_cast<int>(sx.size()) != ctx->mX || static_cast<int>(sz.size()) != ctx->mZ) {
        if (error_out) *error_out = "CUDA input syndrome size mismatch";
        return false;
    }
    if (!time_memcpy(cudaMemcpy(ctx->d_sx, sx.data(), sizeof(int) * sx.size(), cudaMemcpyHostToDevice),
                     error_out, "copy sx", nullptr)) {
        return false;
    }
    if (!time_memcpy(cudaMemcpy(ctx->d_sz, sz.data(), sizeof(int) * sz.size(), cudaMemcpyHostToDevice),
                     error_out, "copy sz", nullptr)) {
        return false;
    }
    if (check_warmup < 0) {
        check_warmup = 0;
    }
    if (check_interval <= 0) {
        check_interval = 1;
    }
    DeviceMsg d_prior = make_msg(static_cast<MsgReal>(prior.v[0]),
                                 static_cast<MsgReal>(prior.v[1]),
                                 static_cast<MsgReal>(prior.v[2]),
                                 static_cast<MsgReal>(prior.v[3]));
    int threads = 256;
    int x_blocks = (ctx->x_edges + threads - 1) / threads;
    int z_blocks = (ctx->z_edges + threads - 1) / threads;
    int var_blocks = (ctx->nvars + threads - 1) / threads;
    int check_x_blocks = (ctx->mX + threads - 1) / threads;
    int check_z_blocks = (ctx->mZ + threads - 1) / threads;
    int check_x_threads = ctx->max_x_deg > 0 ? ctx->max_x_deg : 1;
    int check_z_threads = ctx->max_z_deg > 0 ? ctx->max_z_deg : 1;
    const int prefix_threshold = 32;
    bool use_prefix_x = ctx->max_x_deg >= prefix_threshold;
    bool use_prefix_z = ctx->max_z_deg >= prefix_threshold;
    if (check_x_threads > 256) {
        check_x_threads = 256;
        use_prefix_x = false;
    }
    if (check_z_threads > 256) {
        check_z_threads = 256;
        use_prefix_z = false;
    }
    size_t shared_x_bytes = use_prefix_x ? static_cast<size_t>(ctx->max_x_deg) * 6 * sizeof(MsgReal) : 0;
    size_t shared_z_bytes = use_prefix_z ? static_cast<size_t>(ctx->max_z_deg) * 6 * sizeof(MsgReal) : 0;

    const bool record_costs = measure_costs && !use_graph;
    cudaStream_t stream = use_graph ? ctx->graph_stream : 0;
    if (use_graph && !stream) {
        if (!check_cuda(cudaStreamCreate(&ctx->graph_stream), error_out, "cudaStreamCreate")) return false;
        stream = ctx->graph_stream;
    }

    auto enqueue_iter_kernels = [&](cudaStream_t stream, bool run_x, bool run_z) {
        if (run_x) {
            if (use_prefix_x) {
                check_update_x_by_check_kernel<<<ctx->mX, check_x_threads, shared_x_bytes, stream>>>(
                    ctx->mX,
                    ctx->d_x_check_offsets,
                    ctx->d_x_check_edges,
                    ctx->d_sx,
                    ctx->d_x_v2c,
                    ctx->d_x_c2v
                );
            } else {
                check_update_x_kernel<<<x_blocks, threads, 0, stream>>>(
                    ctx->x_edges,
                    ctx->d_x_check_offsets,
                    ctx->d_x_check_edges,
                    ctx->d_x_edge_check,
                    ctx->d_sx,
                    ctx->d_x_v2c,
                    ctx->d_x_c2v
                );
            }
        }
        if (run_z) {
            if (use_prefix_z) {
                check_update_z_by_check_kernel<<<ctx->mZ, check_z_threads, shared_z_bytes, stream>>>(
                    ctx->mZ,
                    ctx->d_z_check_offsets,
                    ctx->d_z_check_edges,
                    ctx->d_sz,
                    ctx->d_z_v2c,
                    ctx->d_z_c2v
                );
            } else {
                check_update_z_kernel<<<z_blocks, threads, 0, stream>>>(
                    ctx->z_edges,
                    ctx->d_z_check_offsets,
                    ctx->d_z_check_edges,
                    ctx->d_z_edge_check,
                    ctx->d_sz,
                    ctx->d_z_v2c,
                    ctx->d_z_c2v
                );
            }
        }
        variable_update_kernel<<<var_blocks, threads, 0, stream>>>(
            ctx->nvars,
            ctx->d_x_var_offsets,
            ctx->d_x_var_edges,
            ctx->d_z_var_offsets,
            ctx->d_z_var_edges,
            ctx->d_x_c2v,
            ctx->d_z_c2v,
            ctx->d_x_v2c,
            ctx->d_z_v2c,
            d_prior,
            damping,
            ctx->d_freeze_x,
            ctx->d_freeze_z,
            ctx->d_est,
            nullptr,
            nullptr
        );
    };

    auto enqueue_check_kernels = [&](cudaStream_t stream, std::string *err_out) -> bool {
        if (!check_cuda(cudaMemsetAsync(ctx->d_syn_x, 1, sizeof(int), stream), err_out, "memset syn_x")) return false;
        if (!check_cuda(cudaMemsetAsync(ctx->d_syn_z, 1, sizeof(int), stream), err_out, "memset syn_z")) return false;
        syndrome_compare_kernel<<<check_x_blocks, threads, 0, stream>>>(
            ctx->mX,
            ctx->d_x_check_offsets,
            ctx->d_x_check_edges,
            ctx->d_x_edge_var,
            ctx->d_est,
            ctx->d_sx,
            1,
            ctx->d_syn_x
        );
        syndrome_compare_kernel<<<check_z_blocks, threads, 0, stream>>>(
            ctx->mZ,
            ctx->d_z_check_offsets,
            ctx->d_z_check_edges,
            ctx->d_z_edge_var,
            ctx->d_est,
            ctx->d_sz,
            0,
            ctx->d_syn_z
        );
        return true;
    };

    int freeze_init = 0;
    if (!time_memcpy(cudaMemcpy(ctx->d_freeze_x, &freeze_init, sizeof(int), cudaMemcpyHostToDevice),
                     error_out, "copy freeze_x_init", nullptr)) {
        return false;
    }
    if (!time_memcpy(cudaMemcpy(ctx->d_freeze_z, &freeze_init, sizeof(int), cudaMemcpyHostToDevice),
                     error_out, "copy freeze_z_init", nullptr)) {
        return false;
    }

    bool built_graph = false;
    if (use_graph) {
        bool prior_match = true;
        for (int i = 0; i < 4; ++i) {
            if (ctx->graph_prior.v[i] != prior.v[i]) {
                prior_match = false;
                break;
            }
        }
        bool needs_graph = !ctx->graph_ready || !prior_match || (ctx->graph_damping != damping);
        if (needs_graph) {
            if (ctx->iter_exec) {
                cudaGraphExecDestroy(ctx->iter_exec);
                ctx->iter_exec = nullptr;
            }
            if (ctx->check_exec) {
                cudaGraphExecDestroy(ctx->check_exec);
                ctx->check_exec = nullptr;
            }
            if (ctx->iter_graph) {
                cudaGraphDestroy(ctx->iter_graph);
                ctx->iter_graph = nullptr;
            }
            if (ctx->check_graph) {
                cudaGraphDestroy(ctx->check_graph);
                ctx->check_graph = nullptr;
            }
            if (!check_cuda(cudaStreamSynchronize(stream), error_out, "cudaStreamSync before capture")) return false;
            if (!check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), error_out,
                            "cudaStreamBeginCapture iter")) {
                return false;
            }
            enqueue_iter_kernels(stream, true, true);
            if (!check_cuda(cudaStreamEndCapture(stream, &ctx->iter_graph), error_out, "cudaStreamEndCapture iter")) {
                return false;
            }
            if (!check_cuda(cudaGraphInstantiate(&ctx->iter_exec, ctx->iter_graph, nullptr, nullptr, 0), error_out,
                            "cudaGraphInstantiate iter")) {
                return false;
            }

            if (!check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), error_out,
                            "cudaStreamBeginCapture check")) {
                return false;
            }
            enqueue_iter_kernels(stream, true, true);
            if (!enqueue_check_kernels(stream, error_out)) return false;
            if (!check_cuda(cudaStreamEndCapture(stream, &ctx->check_graph), error_out, "cudaStreamEndCapture check")) {
                return false;
            }
            if (!check_cuda(cudaGraphInstantiate(&ctx->check_exec, ctx->check_graph, nullptr, nullptr, 0), error_out,
                            "cudaGraphInstantiate check")) {
                return false;
            }

            ctx->graph_ready = true;
            ctx->graph_prior = prior;
            ctx->graph_damping = damping;
            ctx->graph_builds++;
            built_graph = true;
        } else {
            ctx->graph_reuses++;
        }
    }

    cudaEvent_t kernel_start{};
    cudaEvent_t kernel_stop{};
    if (!check_cuda(cudaEventCreate(&kernel_start), error_out, "cudaEventCreate start")) return false;
    if (!check_cuda(cudaEventCreate(&kernel_stop), error_out, "cudaEventCreate stop")) return false;
    if (!check_cuda(cudaEventRecord(kernel_start, stream), error_out, "cudaEventRecord start")) return false;

    cudaEvent_t init_start{};
    cudaEvent_t init_stop{};
    if (record_costs) {
        if (!check_cuda(cudaEventCreate(&init_start), error_out, "cudaEventCreate init start")) return false;
        if (!check_cuda(cudaEventCreate(&init_stop), error_out, "cudaEventCreate init stop")) return false;
        if (!check_cuda(cudaEventRecord(init_start, stream), error_out, "cudaEventRecord init start")) return false;
    }

    cudaEvent_t check_start{};
    cudaEvent_t check_stop{};
    double check_kernel_ms = 0.0;
    if (record_costs) {
        if (!check_cuda(cudaEventCreate(&check_start), error_out, "cudaEventCreate check start")) return false;
        if (!check_cuda(cudaEventCreate(&check_stop), error_out, "cudaEventCreate check stop")) return false;
    }

    init_messages_kernel<<<x_blocks, threads, 0, stream>>>(ctx->x_edges, ctx->d_x_v2c, ctx->d_x_c2v, d_prior);
    init_messages_kernel<<<z_blocks, threads, 0, stream>>>(ctx->z_edges, ctx->d_z_v2c, ctx->d_z_c2v, d_prior);
    if (!check_cuda(cudaGetLastError(), error_out, "init_messages_kernel")) return false;
    double init_kernel_ms = 0.0;
    if (record_costs) {
        if (!check_cuda(cudaEventRecord(init_stop, stream), error_out, "cudaEventRecord init stop")) return false;
        if (!check_cuda(cudaEventSynchronize(init_stop), error_out, "cudaEventSync init stop")) return false;
        float init_ms = 0.0f;
        if (!check_cuda(cudaEventElapsedTime(&init_ms, init_start, init_stop),
                        error_out, "cudaEventElapsedTime init")) {
            return false;
        }
        init_kernel_ms = static_cast<double>(init_ms);
    }

    bool freeze_x = false;
    bool freeze_z = false;
    bool syn_all = false;
    int last_checked_iter = -1;
    int check_count = 0;
    int iter = 0;
    for (; iter < max_iter; ++iter) {
        bool do_check = (iter + 1 == max_iter);
        if (!do_check && (iter + 1 > check_warmup)) {
            int after_warmup = iter + 1 - check_warmup;
            do_check = (after_warmup % check_interval == 0);
        }
        if (use_graph) {
            cudaGraphExec_t exec = do_check ? ctx->check_exec : ctx->iter_exec;
            if (!check_cuda(cudaGraphLaunch(exec, stream), error_out,
                            do_check ? "cudaGraphLaunch check" : "cudaGraphLaunch iter")) {
                return false;
            }
            if (!check_cuda(cudaGetLastError(), error_out,
                            do_check ? "cudaGraphLaunch check" : "cudaGraphLaunch iter")) {
                return false;
            }
        } else {
            enqueue_iter_kernels(stream, !freeze_x, !freeze_z);
            if (!check_cuda(cudaGetLastError(), error_out, "bp_kernels")) return false;
            if (do_check) {
                if (record_costs) {
                    if (!check_cuda(cudaEventRecord(check_start, stream), error_out, "cudaEventRecord check start")) {
                        return false;
                    }
                }
                if (!enqueue_check_kernels(stream, error_out)) return false;
                if (!check_cuda(cudaGetLastError(), error_out, "syndrome_compare_kernel")) return false;
                if (record_costs) {
                    if (!check_cuda(cudaEventRecord(check_stop, stream), error_out, "cudaEventRecord check stop")) {
                        return false;
                    }
                    if (!check_cuda(cudaEventSynchronize(check_stop), error_out, "cudaEventSync check stop")) {
                        return false;
                    }
                    float check_ms = 0.0f;
                    if (!check_cuda(cudaEventElapsedTime(&check_ms, check_start, check_stop),
                                    error_out, "cudaEventElapsedTime check")) {
                        return false;
                    }
                    check_kernel_ms += static_cast<double>(check_ms);
                }
            }
        }

        if (do_check) {
            int syn_x_flag = 0;
            int syn_z_flag = 0;
            if (!time_memcpy(cudaMemcpy(&syn_x_flag, ctx->d_syn_x, sizeof(int), cudaMemcpyDeviceToHost),
                             error_out, "copy syn_x", &check_memcpy_ms)) return false;
            if (!time_memcpy(cudaMemcpy(&syn_z_flag, ctx->d_syn_z, sizeof(int), cudaMemcpyDeviceToHost),
                             error_out, "copy syn_z", &check_memcpy_ms)) return false;

            bool syn_x = syn_x_flag != 0;
            bool syn_z = syn_z_flag != 0;
            syn_all = syn_x && syn_z;
            last_checked_iter = iter;
            check_count++;
            if (freeze_syn) {
                if (syn_x && !freeze_x) {
                    freeze_x = true;
                    int one = 1;
                    if (!time_memcpy(cudaMemcpy(ctx->d_freeze_x, &one, sizeof(int), cudaMemcpyHostToDevice),
                                     error_out, "copy freeze_x", nullptr)) {
                        return false;
                    }
                    freeze_msgs_kernel<<<x_blocks, threads, 0, stream>>>(
                        ctx->x_edges, ctx->d_x_edge_var, ctx->d_est, 1, ctx->d_x_v2c, ctx->d_x_c2v
                    );
                }
                if (syn_z && !freeze_z) {
                    freeze_z = true;
                    int one = 1;
                    if (!time_memcpy(cudaMemcpy(ctx->d_freeze_z, &one, sizeof(int), cudaMemcpyHostToDevice),
                                     error_out, "copy freeze_z", nullptr)) {
                        return false;
                    }
                    freeze_msgs_kernel<<<z_blocks, threads, 0, stream>>>(
                        ctx->z_edges, ctx->d_z_edge_var, ctx->d_est, 0, ctx->d_z_v2c, ctx->d_z_c2v
                    );
                }
                if (!check_cuda(cudaGetLastError(), error_out, "freeze_msgs_kernel")) return false;
            }
            if (syn_all) {
                break;
            }
        }
    }

    if (last_checked_iter != iter) {
        if (!enqueue_check_kernels(stream, error_out)) return false;
        if (!check_cuda(cudaGetLastError(), error_out, "syndrome_compare_kernel_final")) return false;

        int syn_x_flag = 0;
        int syn_z_flag = 0;
        if (!time_memcpy(cudaMemcpy(&syn_x_flag, ctx->d_syn_x, sizeof(int), cudaMemcpyDeviceToHost),
                         error_out, "copy syn_x_final", &check_memcpy_ms)) return false;
        if (!time_memcpy(cudaMemcpy(&syn_z_flag, ctx->d_syn_z, sizeof(int), cudaMemcpyDeviceToHost),
                         error_out, "copy syn_z_final", &check_memcpy_ms)) return false;
        syn_all = (syn_x_flag != 0) && (syn_z_flag != 0);
        check_count++;
    }

    out.est.assign(ctx->nvars, 0);
    if (!time_memcpy(cudaMemcpy(out.est.data(), ctx->d_est, sizeof(int) * ctx->nvars, cudaMemcpyDeviceToHost),
                     error_out, "copy est", nullptr)) {
        return false;
    }
    out.iterations = iter + 1;
    out.check_count = check_count;
    out.syndrome_match = syn_all;
    float kernel_ms = 0.0f;
    if (!check_cuda(cudaEventRecord(kernel_stop, stream), error_out, "cudaEventRecord stop")) return false;
    if (!check_cuda(cudaEventSynchronize(kernel_stop), error_out, "cudaEventSync stop")) return false;
    if (!check_cuda(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop), error_out, "cudaEventElapsedTime")) {
        return false;
    }
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    if (record_costs) {
        cudaEventDestroy(check_start);
        cudaEventDestroy(check_stop);
        cudaEventDestroy(init_start);
        cudaEventDestroy(init_stop);
    }
    out.kernel_ms = kernel_ms;
    out.memcpy_ms = memcpy_ms;
    out.check_kernel_ms = check_kernel_ms;
    out.check_memcpy_ms = check_memcpy_ms;
    out.init_kernel_ms = init_kernel_ms;
    double total_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - host_start)
                          .count();
    out.host_ms = std::max(0.0, total_ms - memcpy_ms - static_cast<double>(kernel_ms));
    if (use_graph && log_graph) {
        std::cerr << "[cuda-graph] build=" << ctx->graph_builds
                  << " reuse=" << ctx->graph_reuses
                  << " this=" << (built_graph ? "build" : "reuse")
                  << "\n";
    }
    return true;
}

#endif
