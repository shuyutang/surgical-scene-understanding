// D4: end-to-end latency of the C++ runtime per stereo pair.
//   bench <engine> <model_dir> <frames.u8> <n_pairs_in_file> <iterations> <kappa> <out.csv>
// Frames are pre-decoded (video decode is excluded from D4 and measured separately in Python) and
// cycled; each iteration copies one pair into the pinned input buffer, then runs the full pipeline.
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "surgscene/npy.h"
#include "surgscene/pipeline.h"

using namespace surgscene;

static StereoRig load_rig(const std::string& G) {
  auto K = load_npy(G + "/rig_K.npy").as_double(), D = load_npy(G + "/rig_dist.npy").as_double();
  auto R = load_npy(G + "/rig_R.npy").as_double(), T = load_npy(G + "/rig_T.npy").as_double();
  StereoRig r;
  Camera* cams[2] = {&r.left, &r.right};
  for (int c = 0; c < 2; ++c) {
    for (int i = 0; i < 9; ++i) cams[c]->K(i / 3, i % 3) = K[c * 9 + i];
    for (int i = 0; i < 5; ++i) cams[c]->dist(i) = D[c * 5 + i];
  }
  for (int i = 0; i < 9; ++i) r.R(i / 3, i % 3) = R[i];
  for (int i = 0; i < 3; ++i) r.T(i) = T[i];
  return r;
}

static double pct(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, static_cast<size_t>(p / 100.0 * (v.size() - 1) + 0.5))];
}

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "usage: bench <engine> <model_dir> <frames.u8> <n_pairs> <iters> <kappa> <out.csv>\n";
    return 2;
  }
  const std::string engine = argv[1], model_dir = argv[2], frames_path = argv[3], csv = argv[7];
  const int n_pairs = std::stoi(argv[4]), iters = std::stoi(argv[5]);
  const double kappa = std::stod(argv[6]);
  const size_t pair_bytes = TrtEngine::input_bytes();
  std::vector<uint8_t> frames(pair_bytes * n_pairs);
  std::ifstream f(frames_path, std::ios::binary);
  f.read(reinterpret_cast<char*>(frames.data()), frames.size());
  if (!f) {
    std::cerr << "cannot read " << n_pairs << " pairs from " << frames_path << "\n";
    return 1;
  }

  Pipeline pipe(engine, model_dir, load_rig(model_dir), MapParams{}, KFParams{}, kappa);
  PipelineOutput out;
  const int warmup = 30;
  std::vector<double> copy_in, h2d, infer, d2h, map, kalman, tri, total, e2e;
  std::ofstream o(csv);
  o << "iter,copy_in,h2d,infer,d2h,map,kalman,triangulate,pipeline,end_to_end,"
       "tip1_x,tip1_y,tip1_z,tip3_x,tip3_y,tip3_z\n";
  for (int it = 0; it < warmup + iters; ++it) {
    const auto t0 = std::chrono::steady_clock::now();
    std::memcpy(pipe.input(), frames.data() + (it % n_pairs) * pair_bytes, pair_bytes);
    const double c = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    pipe.process(out);
    const double e = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (it < warmup) continue;
    const auto& t = out.times;
    copy_in.push_back(c), h2d.push_back(t.h2d), infer.push_back(t.infer), d2h.push_back(t.d2h), map.push_back(t.map),
        kalman.push_back(t.kalman), tri.push_back(t.triangulate), total.push_back(t.total), e2e.push_back(e);
    o << it - warmup << "," << c << "," << t.h2d << "," << t.infer << "," << t.d2h << "," << t.map << "," << t.kalman
      << "," << t.triangulate << "," << t.total << "," << e;
    for (const auto& tip : out.tips) o << "," << tip.X(0) << "," << tip.X(1) << "," << tip.X(2);
    o << "\n";
  }
  auto row = [&](const char* name, const std::vector<double>& v) {
    std::printf("%-26s p50 %7.3f  p99 %7.3f  max %7.3f ms\n", name, pct(v, 50), pct(v, 99), pct(v, 100));
  };
  std::printf("%d iterations (after %d warmup), stereo pair 2x1400x986\n", iters, warmup);
  row("copy into pinned input", copy_in);
  row("H2D (8.3 MB)", h2d);
  row("TensorRT (pre+net+decode)", infer);
  row("D2H", d2h);
  row("gated MAP x4", map);
  row("Kalman x20", kalman);
  row("triangulation x4", tri);
  row("pipeline total", total);
  row("end to end (incl. copy)", e2e);
  return 0;
}
