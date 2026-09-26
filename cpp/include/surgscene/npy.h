// Minimal .npy reader (little-endian, C order, float64/float32/uint8) for golden files and model data.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace surgscene {

struct NpyArray {
  std::vector<size_t> shape;
  char dtype = 0;  // 'f' (float) or 'u' (unsigned int)
  size_t word = 0; // bytes per element
  std::vector<uint8_t> bytes;

  size_t size() const;
  std::vector<double> as_double() const;  // converts float32/float64/uint8
  const uint8_t* u8() const { return bytes.data(); }
};

NpyArray load_npy(const std::string& path);

}  // namespace surgscene
