#include "surgscene/npy.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace surgscene {

size_t NpyArray::size() const {
  size_t n = 1;
  for (auto s : shape) n *= s;
  return n;
}

std::vector<double> NpyArray::as_double() const {
  std::vector<double> out(size());
  for (size_t i = 0; i < out.size(); ++i) {
    if (dtype == 'f' && word == 8) {
      double v;
      std::memcpy(&v, bytes.data() + 8 * i, 8);
      out[i] = v;
    } else if (dtype == 'f' && word == 4) {
      float v;
      std::memcpy(&v, bytes.data() + 4 * i, 4);
      out[i] = v;
    } else if (dtype == 'u' && word == 1) {
      out[i] = bytes[i];
    } else {
      throw std::runtime_error("npy: unsupported dtype");
    }
  }
  return out;
}

NpyArray load_npy(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("npy: cannot open " + path);
  char magic[6];
  f.read(magic, 6);
  if (std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error("npy: bad magic " + path);
  uint8_t ver[2];
  f.read(reinterpret_cast<char*>(ver), 2);
  uint32_t hlen = 0;
  if (ver[0] == 1) {
    uint16_t h;
    f.read(reinterpret_cast<char*>(&h), 2);
    hlen = h;
  } else {
    f.read(reinterpret_cast<char*>(&hlen), 4);
  }
  std::string header(hlen, ' ');
  f.read(header.data(), hlen);
  if (header.find("'fortran_order': False") == std::string::npos) throw std::runtime_error("npy: fortran order");
  NpyArray a;
  auto d = header.find("'descr': '");
  const char endian = header[d + 10];
  if (endian == '>') throw std::runtime_error("npy: big endian");
  a.dtype = header[d + 11];
  a.word = std::stoul(header.substr(d + 12, header.find('\'', d + 12) - d - 12));
  auto s0 = header.find('(', header.find("'shape'"));
  auto s1 = header.find(')', s0);
  std::string sh = header.substr(s0 + 1, s1 - s0 - 1);
  size_t pos = 0;
  while (pos < sh.size()) {
    auto comma = sh.find(',', pos);
    std::string tok = sh.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
    if (tok.find_first_of("0123456789") != std::string::npos) a.shape.push_back(std::stoul(tok));
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  a.bytes.resize(a.size() * a.word);
  f.read(reinterpret_cast<char*>(a.bytes.data()), a.bytes.size());
  if (!f) throw std::runtime_error("npy: truncated " + path);
  return a;
}

}  // namespace surgscene
