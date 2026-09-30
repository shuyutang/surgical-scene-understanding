// Arm C: the same engine run inside a Holoscan application (C++), InferenceOp with the TensorRT backend.
//   bench_holoscan <engine.plan> <frames.u8> <n_pairs> <iters> <scheduler: greedy|multithread> <cuda_graphs: 0|1> <out_prefix>
// Graph: Source -> InferenceOp -> Sink.
//   Source   copies the pair from host memory into a device tensor from a BlockMemoryPool (H2D), emits it
//   Infer    InferenceOp: input on GPU, outputs copied to host (output_on_cuda false), own CUDA stream
//   Sink     receives kp / conf / sigma in host memory, stops the clock
// Closed loop, as in the other arms: the graph is a cycle. The Sink sends an ack back to the Source,
// which emits the next pair only after it (the first pair needs none), so exactly one pair is in
// flight. A BooleanCondition gate stops the Source after the last pair. Latency = Sink done - Source
// start, per pair, so it includes the framework's scheduling and message passing between operators.
// Schedulers: greedy (one thread, the default) or multithread (2 workers, condition polling with no
// recession period). A gate re-opened by another operator (instead of the ack cycle) stalls the
// multi-threaded schedulers: a disabled BooleanCondition reports NEVER and the operator is dropped.
#include <cuda_runtime_api.h>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/inference/inference.hpp>
#include <iostream>
#include <memory>

#include "bench_common.h"

namespace {

struct State {
  std::vector<uint8_t> frames;
  int n_pairs = 0;
  int it = 0;
  bench::Clock::time_point t0;
  std::unique_ptr<bench::Recorder> rec;
  std::shared_ptr<holoscan::BooleanCondition> gate;
};
State g;

void cuda_ok(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

}  // namespace

namespace holoscan::ops {

class SourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SourceOp)
  SourceOp() = default;
  void setup(OperatorSpec& spec) override {
    spec.input<int>("ack").condition(ConditionType::kNone);  // polled; the first pair needs no ack
    spec.output<gxf::Entity>("out");
    spec.param(allocator_, "allocator", "Allocator", "device memory for the input tensor");
  }
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) override {
    auto ack = op_input.receive<int>("ack");
    if (!first_ && !ack) return;  // previous pair still in flight
    first_ = false;
    g.t0 = bench::Clock::now();
    const uint8_t* pair = g.frames.data() + size_t(g.it % g.n_pairs) * bench::kPairBytes;
    auto entity = gxf::Entity::New(&context);
    auto alloc = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());
    auto t = static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>("frames").value();
    t->reshape<uint8_t>(nvidia::gxf::Shape({bench::kImages, bench::kH, bench::kW, 3}),
                        nvidia::gxf::MemoryStorageType::kDevice, alloc.value());
    cuda_ok(cudaMemcpy(t->pointer(), pair, bench::kPairBytes, cudaMemcpyHostToDevice), "h2d");
    // from pageable memory, cudaMemcpy may return before the DMA has finished, and InferenceOp reads the
    // tensor on its own stream, which isn't ordered after this copy: wait for it
    cuda_ok(cudaDeviceSynchronize(), "sync h2d");
    op_output.emit(entity, "out");
    if (++emitted_ == g.rec->total_iters()) g.gate->disable_tick();  // last pair: stop the Source
  }

 private:
  Parameter<std::shared_ptr<Allocator>> allocator_;
  bool first_ = true;
  int emitted_ = 0;
};

class SinkOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SinkOp)
  SinkOp() = default;
  void setup(OperatorSpec& spec) override {
    spec.input<gxf::Entity>("in");
    spec.output<int>("ack");
  }
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
    auto msg = op_input.receive<gxf::Entity>("in").value();
    bench::Outputs o;
    size_t off = 0;
    for (const char* n : {"kp", "conf", "sigma"}) {
      auto t = msg.get<Tensor>(n);
      if (!t) throw std::runtime_error(std::string("missing output tensor ") + n);
      std::memcpy(reinterpret_cast<uint8_t*>(o.v) + off, t->data(), t->nbytes());
      off += t->nbytes();
    }
    const double ms = bench::ms_since(g.t0);
    if (off != sizeof(o.v)) throw std::runtime_error("unexpected output size " + std::to_string(off));
    g.rec->add(g.it, ms, o);
    ++g.it;
    op_output.emit(g.it, "ack");
  }
};

}  // namespace holoscan::ops

class BenchApp : public holoscan::Application {
 public:
  BenchApp(std::string engine, bool cuda_graphs) : engine_(std::move(engine)), cuda_graphs_(cuda_graphs) {}
  void compose() override {
    using namespace holoscan;
    auto pool = make_resource<BlockMemoryPool>("pool", Arg("storage_type") = int32_t(1),  // device
                                               Arg("block_size") = uint64_t(bench::kPairBytes),
                                               Arg("num_blocks") = uint64_t(4));
    auto infer_alloc = make_resource<UnboundedAllocator>("infer_alloc");
    auto streams = make_resource<CudaStreamPool>("streams", 0, 0, 0, 1, 2);
    g.gate = make_condition<BooleanCondition>("gate", true);
    auto source = make_operator<ops::SourceOp>("source", Arg("allocator") = pool, g.gate);
    ops::InferenceOp::DataMap model_path_map;
    model_path_map.insert("kp_stereo", engine_);
    ops::InferenceOp::DataVecMap pre_map, infer_map;
    pre_map.insert("kp_stereo", {"frames"});
    infer_map.insert("kp_stereo", {"kp", "conf", "sigma"});
    auto infer = make_operator<ops::InferenceOp>(
        "infer", Arg("backend") = std::string("trt"), Arg("model_path_map") = model_path_map,
        Arg("pre_processor_map") = pre_map, Arg("inference_map") = infer_map,
        Arg("in_tensor_names") = std::vector<std::string>{"frames"},
        Arg("out_tensor_names") = std::vector<std::string>{"kp", "conf", "sigma"},
        Arg("is_engine_path") = true, Arg("input_on_cuda") = true, Arg("output_on_cuda") = false,
        Arg("transmit_on_cuda") = false, Arg("enable_cuda_graphs") = cuda_graphs_,
        Arg("allocator") = infer_alloc, Arg("cuda_stream_pool") = streams);
    auto sink = make_operator<ops::SinkOp>("sink");
    add_flow(source, infer, {{"out", "receivers"}});
    add_flow(infer, sink, {{"transmitter", "in"}});
    add_flow(sink, source, {{"ack", "ack"}});
  }

 private:
  std::string engine_;
  bool cuda_graphs_;
};

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "usage: bench_holoscan <engine> <frames.u8> <n_pairs> <iters> greedy|multithread 0|1 <out_prefix>\n";
    return 2;
  }
  const std::string sched = argv[5];
  const bool graphs = std::string(argv[6]) == "1";
  g.n_pairs = std::stoi(argv[3]);
  g.frames = bench::load_frames(argv[2], g.n_pairs);
  g.rec = std::make_unique<bench::Recorder>("Holoscan InferenceOp (" + sched + (graphs ? ", CUDA graphs)" : ")"),
                                            g.n_pairs, std::stoi(argv[4]));
  auto app = holoscan::make_application<BenchApp>(argv[1], graphs);
  if (sched == "multithread") {
    app->scheduler(app->make_scheduler<holoscan::MultiThreadScheduler>(
        "multithread", holoscan::Arg("worker_thread_number") = int64_t(2),
        holoscan::Arg("check_recession_period_ms") = 0.0));
  } else {
    app->scheduler(app->make_scheduler<holoscan::GreedyScheduler>("greedy"));
  }
  app->run();
  if (g.it != g.rec->total_iters()) {
    std::cerr << "only " << g.it << " of " << g.rec->total_iters() << " iterations completed\n";
    return 1;
  }
  g.rec->finish(argv[7]);
  return 0;
}
