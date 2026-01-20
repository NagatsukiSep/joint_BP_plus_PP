#include "jointbp_cuda.h"

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cmath>
#include <sstream>

struct DeviceMsg {
    double v0;
    double v1;
    double v2;
    double v3;
};

__host__ __device__ DeviceMsg make_msg(double a, double b, double c, double d) {
    DeviceMsg m;
    m.v0 = a;
    m.v1 = b;
    m.v2 = c;
    m.v3 = d;
    return m;
}

namespace {

__device__ void normalize_msg(DeviceMsg &m) {
    double sum = m.v0 + m.v1 + m.v2 + m.v3;
    if (sum <= 0.0) {
        m.v0 = 0.25;
        m.v1 = 0.25;
        m.v2 = 0.25;
        m.v3 = 0.25;
        return;
    }
    double inv = 1.0 / sum;
    m.v0 *= inv;
    m.v1 *= inv;
    m.v2 *= inv;
    m.v3 *= inv;
}

__device__ DeviceMsg multiply_msg(const DeviceMsg &a, const DeviceMsg &b) {
    return make_msg(a.v0 * b.v0, a.v1 * b.v1, a.v2 * b.v2, a.v3 * b.v3);
}

__device__ int xbit(int state) {
    return (state == 1 || state == 3) ? 1 : 0;
}

__device__ int zbit(int state) {
    return (state == 2 || state == 3) ? 1 : 0;
}

__device__ DeviceMsg det_msg_xbit(int bit) {
    return bit ? make_msg(0.0, 0.5, 0.0, 0.5) : make_msg(0.5, 0.0, 0.5, 0.0);
}

__device__ DeviceMsg det_msg_zbit(int bit) {
    return bit ? make_msg(0.0, 0.0, 0.5, 0.5) : make_msg(0.5, 0.5, 0.0, 0.0);
}

__global__ void init_messages_kernel(int edges, DeviceMsg *v2c, DeviceMsg *c2v, DeviceMsg prior) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= edges) return;
    v2c[idx] = prior;
    c2v[idx] = make_msg(0.25, 0.25, 0.25, 0.25);
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
    double p_even = 1.0;
    double p_odd = 0.0;
    for (int idx = start; idx < end; ++idx) {
        int ej = check_edges[idx];
        if (ej == e) continue;
        DeviceMsg m = v2c[ej];
        double q0 = m.v0 + m.v2;
        double q1 = m.v1 + m.v3;
        double new_even = p_even * q0 + p_odd * q1;
        double new_odd = p_even * q1 + p_odd * q0;
        p_even = new_even;
        p_odd = new_odd;
    }
    double val0 = (sx[c] == 0) ? p_even : p_odd;
    double val1 = (sx[c] == 0) ? p_odd : p_even;
    DeviceMsg out = make_msg(val0, val1, val0, val1);
    normalize_msg(out);
    c2v[e] = out;
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
    double p_even = 1.0;
    double p_odd = 0.0;
    for (int idx = start; idx < end; ++idx) {
        int ej = check_edges[idx];
        if (ej == e) continue;
        DeviceMsg m = v2c[ej];
        double q0 = m.v0 + m.v1;
        double q1 = m.v2 + m.v3;
        double new_even = p_even * q0 + p_odd * q1;
        double new_odd = p_even * q1 + p_odd * q0;
        p_even = new_even;
        p_odd = new_odd;
    }
    double val0 = (sz[c] == 0) ? p_even : p_odd;
    double val1 = (sz[c] == 0) ? p_odd : p_even;
    DeviceMsg out = make_msg(val0, val0, val1, val1);
    normalize_msg(out);
    c2v[e] = out;
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
    int freeze_x,
    int freeze_z,
    int *est,
    double *abs_llr_x,
    double *abs_llr_z
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nvars) return;
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
    double best_val = total.v0;
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
        double px1 = total.v1 + total.v3;
        double px0 = total.v0 + total.v2;
        double pz1 = total.v2 + total.v3;
        double pz0 = total.v0 + total.v1;
        double llr_x = log(fmax(px1, eps)) - log(fmax(px0, eps));
        double llr_z = log(fmax(pz1, eps)) - log(fmax(pz0, eps));
        abs_llr_x[v] = fabs(llr_x);
        abs_llr_z[v] = fabs(llr_z);
    }
    if (!freeze_x) {
        for (int idx = x_start; idx < x_end; ++idx) {
            int e = x_var_edges[idx];
            DeviceMsg out = prior;
            for (int idx2 = x_start; idx2 < x_end; ++idx2) {
                int e2 = x_var_edges[idx2];
                if (e2 == e) continue;
                out = multiply_msg(out, x_c2v[e2]);
            }
            for (int idx2 = z_start; idx2 < z_end; ++idx2) {
                out = multiply_msg(out, z_c2v[z_var_edges[idx2]]);
            }
            normalize_msg(out);
            DeviceMsg old = x_v2c[e];
            out.v0 = (1.0 - damping) * out.v0 + damping * old.v0;
            out.v1 = (1.0 - damping) * out.v1 + damping * old.v1;
            out.v2 = (1.0 - damping) * out.v2 + damping * old.v2;
            out.v3 = (1.0 - damping) * out.v3 + damping * old.v3;
            normalize_msg(out);
            x_v2c[e] = out;
        }
    }
    if (!freeze_z) {
        for (int idx = z_start; idx < z_end; ++idx) {
            int e = z_var_edges[idx];
            DeviceMsg out = prior;
            for (int idx2 = x_start; idx2 < x_end; ++idx2) {
                out = multiply_msg(out, x_c2v[x_var_edges[idx2]]);
            }
            for (int idx2 = z_start; idx2 < z_end; ++idx2) {
                int e2 = z_var_edges[idx2];
                if (e2 == e) continue;
                out = multiply_msg(out, z_c2v[e2]);
            }
            normalize_msg(out);
            DeviceMsg old = z_v2c[e];
            out.v0 = (1.0 - damping) * out.v0 + damping * old.v0;
            out.v1 = (1.0 - damping) * out.v1 + damping * old.v1;
            out.v2 = (1.0 - damping) * out.v2 + damping * old.v2;
            out.v3 = (1.0 - damping) * out.v3 + damping * old.v3;
            normalize_msg(out);
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

__global__ void mismatch_count_kernel(int n, const int *a, const int *b, int *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (a[idx] != b[idx]) {
        atomicAdd(out, 1);
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

    if (!check_cuda(cudaMalloc(reinterpret_cast<void **>(&ctx->d_x_v2c), sizeof(DeviceMsg) * graph.x_edges), error_out, "x_v2c")) return nullptr;
    if (!check_cuda(cudaMalloc(reinterpret_cast<void **>(&ctx->d_x_c2v), sizeof(DeviceMsg) * graph.x_edges), error_out, "x_c2v")) return nullptr;
    if (!check_cuda(cudaMalloc(reinterpret_cast<void **>(&ctx->d_z_v2c), sizeof(DeviceMsg) * graph.z_edges), error_out, "z_v2c")) return nullptr;
    if (!check_cuda(cudaMalloc(reinterpret_cast<void **>(&ctx->d_z_c2v), sizeof(DeviceMsg) * graph.z_edges), error_out, "z_c2v")) return nullptr;

    return ctx;
}

void cuda_bp_destroy(CudaBPContext *ctx) {
    if (!ctx) return;
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
    bool freeze_syn,
    double damping,
    CudaBPResult &out,
    std::string *error_out
) {
    if (!ctx) {
        if (error_out) *error_out = "CUDA context not initialized";
        return false;
    }
    if (static_cast<int>(sx.size()) != ctx->mX || static_cast<int>(sz.size()) != ctx->mZ) {
        if (error_out) *error_out = "CUDA input syndrome size mismatch";
        return false;
    }
    if (!check_cuda(cudaMemcpy(ctx->d_sx, sx.data(), sizeof(int) * sx.size(), cudaMemcpyHostToDevice), error_out, "copy sx")) {
        return false;
    }
    if (!check_cuda(cudaMemcpy(ctx->d_sz, sz.data(), sizeof(int) * sz.size(), cudaMemcpyHostToDevice), error_out, "copy sz")) {
        return false;
    }
    DeviceMsg d_prior = make_msg(prior.v[0], prior.v[1], prior.v[2], prior.v[3]);
    int threads = 256;
    int x_blocks = (ctx->x_edges + threads - 1) / threads;
    int z_blocks = (ctx->z_edges + threads - 1) / threads;
    int var_blocks = (ctx->nvars + threads - 1) / threads;
    int check_x_blocks = (ctx->mX + threads - 1) / threads;
    int check_z_blocks = (ctx->mZ + threads - 1) / threads;

    init_messages_kernel<<<x_blocks, threads>>>(ctx->x_edges, ctx->d_x_v2c, ctx->d_x_c2v, d_prior);
    init_messages_kernel<<<z_blocks, threads>>>(ctx->z_edges, ctx->d_z_v2c, ctx->d_z_c2v, d_prior);
    if (!check_cuda(cudaGetLastError(), error_out, "init_messages_kernel")) return false;

    bool freeze_x = false;
    bool freeze_z = false;
    bool syn_all = false;
    int iter = 0;
    for (; iter < max_iter; ++iter) {
        if (!freeze_x) {
            check_update_x_kernel<<<x_blocks, threads>>>(
                ctx->x_edges,
                ctx->d_x_check_offsets,
                ctx->d_x_check_edges,
                ctx->d_x_edge_check,
                ctx->d_sx,
                ctx->d_x_v2c,
                ctx->d_x_c2v
            );
        }
        if (!freeze_z) {
            check_update_z_kernel<<<z_blocks, threads>>>(
                ctx->z_edges,
                ctx->d_z_check_offsets,
                ctx->d_z_check_edges,
                ctx->d_z_edge_check,
                ctx->d_sz,
                ctx->d_z_v2c,
                ctx->d_z_c2v
            );
        }
        variable_update_kernel<<<var_blocks, threads>>>(
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
            freeze_x ? 1 : 0,
            freeze_z ? 1 : 0,
            ctx->d_est,
            nullptr,
            nullptr
        );
        if (!check_cuda(cudaGetLastError(), error_out, "bp_kernels")) return false;

        syndrome_kernel<<<check_x_blocks, threads>>>(
            ctx->mX,
            ctx->d_x_check_offsets,
            ctx->d_x_check_edges,
            ctx->d_x_edge_var,
            ctx->d_est,
            1,
            ctx->d_sx_hat
        );
        syndrome_kernel<<<check_z_blocks, threads>>>(
            ctx->mZ,
            ctx->d_z_check_offsets,
            ctx->d_z_check_edges,
            ctx->d_z_edge_var,
            ctx->d_est,
            0,
            ctx->d_sz_hat
        );
        if (!check_cuda(cudaGetLastError(), error_out, "syndrome_kernel")) return false;

        cudaMemset(ctx->d_mismatch_x, 0, sizeof(int));
        cudaMemset(ctx->d_mismatch_z, 0, sizeof(int));
        mismatch_count_kernel<<<check_x_blocks, threads>>>(ctx->mX, ctx->d_sx_hat, ctx->d_sx, ctx->d_mismatch_x);
        mismatch_count_kernel<<<check_z_blocks, threads>>>(ctx->mZ, ctx->d_sz_hat, ctx->d_sz, ctx->d_mismatch_z);
        if (!check_cuda(cudaGetLastError(), error_out, "mismatch_count_kernel")) return false;

        int mismatch_x = 0;
        int mismatch_z = 0;
        if (!check_cuda(cudaMemcpy(&mismatch_x, ctx->d_mismatch_x, sizeof(int), cudaMemcpyDeviceToHost), error_out, "copy mismatch_x")) return false;
        if (!check_cuda(cudaMemcpy(&mismatch_z, ctx->d_mismatch_z, sizeof(int), cudaMemcpyDeviceToHost), error_out, "copy mismatch_z")) return false;

        bool syn_x = (mismatch_x == 0);
        bool syn_z = (mismatch_z == 0);
        syn_all = syn_x && syn_z;
        if (freeze_syn) {
            if (syn_x && !freeze_x) {
                freeze_x = true;
                freeze_msgs_kernel<<<x_blocks, threads>>>(ctx->x_edges, ctx->d_x_edge_var, ctx->d_est, 1, ctx->d_x_v2c, ctx->d_x_c2v);
            }
            if (syn_z && !freeze_z) {
                freeze_z = true;
                freeze_msgs_kernel<<<z_blocks, threads>>>(ctx->z_edges, ctx->d_z_edge_var, ctx->d_est, 0, ctx->d_z_v2c, ctx->d_z_c2v);
            }
            if (!check_cuda(cudaGetLastError(), error_out, "freeze_msgs_kernel")) return false;
        }
        if (syn_all) {
            break;
        }
    }

    out.est.assign(ctx->nvars, 0);
    if (!check_cuda(cudaMemcpy(out.est.data(), ctx->d_est, sizeof(int) * ctx->nvars, cudaMemcpyDeviceToHost), error_out, "copy est")) {
        return false;
    }
    out.iterations = iter + 1;
    out.syndrome_match = syn_all;
    return true;
}

#endif
