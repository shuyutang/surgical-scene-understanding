// Arm B: the same engine served by Triton Inference Server (TensorRT backend), called from C++ over
// gRPC on localhost, three ways of passing the 8.3 MB input:
//   grpc      the pair is sent inside the gRPC request (the default path)
//   sysshm    the pair is copied into a POSIX shared-memory region registered with the server
//   cudashm   the pair is copied (H2D) into a GPU buffer shared with the server by CUDA IPC; the server
//             reads it without a host copy (Triton's lowest-overhead input path)
// Outputs (320 bytes) come back in the gRPC response in every mode.
//   bench_triton <url> <model> <frames.u8> <n_pairs> <iters> <mode> <out_prefix>
#include <cuda_runtime_api.h>
#include <unistd.h>

#include <iostream>
#include <memory>

#include "bench_common.h"
#include "grpc_client.h"
#include "shm_utils.h"

namespace tc = triton::client;

static void ok(const tc::Error& e, const char* what) {
  if (!e.IsOk()) throw std::runtime_error(std::string(what) + ": " + e.Message());
}
static void cuda_ok(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "usage: bench_triton <url> <model> <frames.u8> <n_pairs> <iters> grpc|sysshm|cudashm <out_prefix>\n";
    return 2;
  }
  const std::string url = argv[1], model = argv[2], mode = argv[6];
  const int n_pairs = std::stoi(argv[4]), iters = std::stoi(argv[5]);
  auto frames = bench::load_frames(argv[3], n_pairs);

  std::unique_ptr<tc::InferenceServerGrpcClient> client;
  ok(tc::InferenceServerGrpcClient::Create(&client, url, false), "client");
  client->UnregisterSystemSharedMemory();
  client->UnregisterCudaSharedMemory();

  tc::InferInput* in_raw;
  ok(tc::InferInput::Create(&in_raw, "frames", {bench::kImages, bench::kH, bench::kW, 3}, "UINT8"), "input");
  std::unique_ptr<tc::InferInput> input(in_raw);
  std::vector<std::unique_ptr<tc::InferRequestedOutput>> outs;
  std::vector<const tc::InferRequestedOutput*> out_ptrs;
  for (const char* n : {"kp", "conf", "sigma"}) {
    tc::InferRequestedOutput* o;
    ok(tc::InferRequestedOutput::Create(&o, n), "output");
    outs.emplace_back(o);
    out_ptrs.push_back(o);
  }
  tc::InferOptions options(model);

  // input path setup (outside the timed loop)
  uint8_t* shm_addr = nullptr;
  int shm_fd = -1;
  void* d_buf = nullptr;
  // region names are unique per process: re-registering a name another client used can be rejected
  const std::string tag = std::to_string(getpid());
  const std::string shm_key = "/surgscene_frames_" + tag, shm_name = "frames_shm_" + tag, cuda_name = "frames_cuda_" + tag;
  if (mode == "sysshm") {
    ok(tc::CreateSharedMemoryRegion(shm_key, bench::kPairBytes, &shm_fd), "create shm");
    ok(tc::MapSharedMemory(shm_fd, 0, bench::kPairBytes, reinterpret_cast<void**>(&shm_addr)), "map shm");
    ok(client->RegisterSystemSharedMemory(shm_name, shm_key, bench::kPairBytes), "register shm");
    ok(input->SetSharedMemory(shm_name, bench::kPairBytes, 0), "set shm");
  } else if (mode == "cudashm") {
    cuda_ok(cudaMalloc(&d_buf, bench::kPairBytes), "cudaMalloc");
    cudaIpcMemHandle_t h;
    cuda_ok(cudaIpcGetMemHandle(&h, d_buf), "ipc handle");
    ok(client->RegisterCudaSharedMemory(cuda_name, h, 0, bench::kPairBytes), "register cuda shm");
    ok(input->SetSharedMemory(cuda_name, bench::kPairBytes, 0), "set cuda shm");
  } else if (mode != "grpc") {
    throw std::runtime_error("unknown mode " + mode);
  }

  bench::Recorder rec("Triton gRPC (" + mode + ")", n_pairs, iters);
  bench::Outputs o;
  for (int it = 0; it < rec.total_iters(); ++it) {
    const uint8_t* pair = frames.data() + size_t(it % n_pairs) * bench::kPairBytes;
    const auto t0 = bench::Clock::now();
    if (mode == "grpc") {
      input->Reset();
      ok(input->AppendRaw(pair, bench::kPairBytes), "append");
    } else if (mode == "sysshm") {
      std::memcpy(shm_addr, pair, bench::kPairBytes);
    } else {
      cuda_ok(cudaMemcpy(d_buf, pair, bench::kPairBytes, cudaMemcpyHostToDevice), "h2d");
      // from pageable memory, cudaMemcpy may return before the DMA into d_buf has finished; the server
      // reads d_buf from another process, so wait for it (without this, outputs drift: a race)
      cuda_ok(cudaDeviceSynchronize(), "sync h2d");
    }
    tc::InferResult* r;
    ok(client->Infer(&r, options, {input.get()}, out_ptrs), "infer");
    std::unique_ptr<tc::InferResult> result(r);
    ok(result->RequestStatus(), "request");
    size_t off = 0;
    for (const char* n : {"kp", "conf", "sigma"}) {
      const uint8_t* buf;
      size_t bytes;
      ok(result->RawData(n, &buf, &bytes), "raw");
      std::memcpy(reinterpret_cast<uint8_t*>(o.v) + off, buf, bytes);
      off += bytes;
    }
    const double ms = bench::ms_since(t0);
    if (off != sizeof(o.v)) throw std::runtime_error("unexpected output size " + std::to_string(off));
    rec.add(it, ms, o);
  }
  rec.finish(argv[7]);

  if (mode == "sysshm") {
    client->UnregisterSystemSharedMemory(shm_name);
    tc::UnmapSharedMemory(shm_addr, bench::kPairBytes);
    tc::CloseSharedMemory(shm_fd);
    tc::UnlinkSharedMemoryRegion(shm_key);
  } else if (mode == "cudashm") {
    client->UnregisterCudaSharedMemory(cuda_name);
    cudaFree(d_buf);
  }
  return 0;
}
