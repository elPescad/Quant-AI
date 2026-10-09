#ifndef NATIVE_MODEL_HPP
#define NATIVE_MODEL_HPP

// Dependency-free inference for the models exported by python/native_model.py.
// Replaces libtorch: the weights are read once into flat arrays and the GRU runs as
// plain vectorisable loops, with no interpreter, dispatcher or tensor allocation.
//
// File layout (little endian):
//   "QGRU" | int32 version | int32 combine | int32 n_nets
//   per net: int32 n_inputs, hidden, layers
//            per layer: weight_ih [3H x in], weight_hh [3H x H], bias_ih [3H], bias_hh [3H]
//                       (PyTorch gate order r, z, n)
//            head weight [3 x (H + n_inputs)], head bias [3]
//            calibration: float inv_temperature, float bias[3]
// Each net reads the first n_inputs features of every window row and outputs
//   logits = head([h_last, x_last]) * inv_temperature + bias.
// combine: 0 single net | 1 mean of the nets' probabilities |
//          2 net0 probabilities, HOLD when net1 disagrees on direction |
//          3 same, but only net0 SELL calls need agreement

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

class NativeModel {
public:
    enum Combine : int32_t { kSingle = 0, kMean = 1, kVeto = 2, kVetoShort = 3 };

    static NativeModel load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("cannot open model " + path);
        char magic[4];
        in.read(magic, 4);
        if (!in || std::memcmp(magic, "QGRU", 4) != 0) throw std::runtime_error(path + " is not a QGRU model file");
        NativeModel m;
        const int32_t version = read_i32(in);
        if (version != 1) throw std::runtime_error(path + ": unsupported model version " + std::to_string(version));
        m.combine_ = static_cast<Combine>(read_i32(in));
        const int32_t n_nets = read_i32(in);
        const int32_t expected_nets = m.combine_ == kSingle ? 1 : 2;
        if (m.combine_ < kSingle || m.combine_ > kVetoShort || n_nets != expected_nets)
            throw std::runtime_error(path + ": bad combine mode / net count");
        for (int k = 0; k < n_nets; ++k) m.nets_.push_back(Net::read(in, path));
        size_t scratch = 0;
        for (const auto& n : m.nets_) {
            m.input_dim_ = std::max(m.input_dim_, n.n_inputs);
            scratch = std::max(scratch, static_cast<size_t>(n.hidden) * (6 + n.layers.size())); // gi, gh, h per layer
        }
        m.scratch_.assign(scratch, 0.0f);
        return m;
    }

    int input_dim() const { return input_dim_; }

    // window: seq_len rows of row_stride floats (oldest first). Writes SELL/HOLD/BUY probabilities.
    void predict(const float* window, int seq_len, int row_stride, float probs[3]) {
        float p0[3], p1[3];
        nets_[0].probs(window, seq_len, row_stride, scratch_.data(), p0);
        if (combine_ == kSingle) {
            std::memcpy(probs, p0, sizeof(p0));
            return;
        }
        nets_[1].probs(window, seq_len, row_stride, scratch_.data(), p1);
        if (combine_ == kMean) {
            for (int c = 0; c < 3; ++c) probs[c] = 0.5f * (p0[c] + p1[c]);
            return;
        }
        const float e0 = p0[2] - p0[0];
        const float e1 = p1[2] - p1[0];
        const bool veto = e0 * e1 < 0.0f && (combine_ == kVeto || e0 < 0.0f);
        if (veto) {
            probs[0] = 0.0f; probs[1] = 1.0f; probs[2] = 0.0f;
        } else {
            std::memcpy(probs, p0, sizeof(p0));
        }
    }

private:
    struct Layer {
        int in = 0;
        std::vector<float> w_ih_t; // [in][3H]  (transposed: contiguous over gate outputs)
        std::vector<float> w_hh_t; // [H][3H]
        std::vector<float> b_ih, b_hh;
    };

    struct Net {
        int n_inputs = 0, hidden = 0;
        std::vector<Layer> layers;
        std::vector<float> head_w_t; // [H + n_inputs][3]
        float head_b[3] = {0, 0, 0};
        float inv_t = 1.0f;
        float cal_b[3] = {0, 0, 0};

        static Net read(std::ifstream& in, const std::string& path) {
            Net n;
            n.n_inputs = read_i32(in);
            n.hidden = read_i32(in);
            const int n_layers = read_i32(in);
            if (n.n_inputs <= 0 || n.hidden <= 0 || n_layers <= 0 || n.n_inputs > 4096 || n.hidden > 4096 || n_layers > 16)
                throw std::runtime_error(path + ": implausible network shape");
            const int g = 3 * n.hidden;
            for (int l = 0; l < n_layers; ++l) {
                Layer L;
                L.in = l == 0 ? n.n_inputs : n.hidden;
                L.w_ih_t = transpose(read_f32(in, static_cast<size_t>(g) * L.in), g, L.in);
                L.w_hh_t = transpose(read_f32(in, static_cast<size_t>(g) * n.hidden), g, n.hidden);
                L.b_ih = read_f32(in, g);
                L.b_hh = read_f32(in, g);
                n.layers.push_back(std::move(L));
            }
            n.head_w_t = transpose(read_f32(in, static_cast<size_t>(3) * (n.hidden + n.n_inputs)), 3, n.hidden + n.n_inputs);
            const auto hb = read_f32(in, 3);
            std::memcpy(n.head_b, hb.data(), sizeof(n.head_b));
            n.inv_t = read_f32(in, 1)[0];
            const auto cb = read_f32(in, 3);
            std::memcpy(n.cal_b, cb.data(), sizeof(n.cal_b));
            if (!in) throw std::runtime_error(path + ": truncated model file");
            return n;
        }

        // out[0..rows) = b + W^T-stored matrix times x (x has `cols` entries)
        static void affine(const float* w_t, const float* b, const float* x, int cols, int rows, float* out) {
            std::memcpy(out, b, sizeof(float) * rows);
            for (int c = 0; c < cols; ++c) {
                const float xc = x[c];
                const float* w = w_t + static_cast<size_t>(c) * rows;
                for (int r = 0; r < rows; ++r) out[r] += xc * w[r];
            }
        }

        // exp(x) = 2^k e^r, |r| <= ln2/2, degree-7 Taylor polynomial (rel. error ~1e-7, i.e. float
        // rounding). Branch-free, so the gate loops vectorise without -ffast-math, which would
        // also let the compiler drop the NaN checks the rest of the engine relies on.
        static float vexp(float x) {
            x = std::min(std::max(x, -87.0f), 88.0f);
            const float k = std::floor(x * 1.44269504f + 0.5f);
            const float r = (x - k * 0.693145752f) - k * 1.42860677e-6f;
            float p = 1.0f / 5040.0f;
            p = p * r + 1.0f / 720.0f;
            p = p * r + 1.0f / 120.0f;
            p = p * r + 1.0f / 24.0f;
            p = p * r + 1.0f / 6.0f;
            p = p * r + 0.5f;
            p = p * r + 1.0f;
            p = p * r + 1.0f;
            return p * std::bit_cast<float>((static_cast<int32_t>(k) + 127) << 23);
        }
        static float sigmoid(float v) { return 1.0f / (1.0f + vexp(-v)); }
        static float vtanh(float v) { return 1.0f - 2.0f / (1.0f + vexp(2.0f * v)); }

        void probs(const float* window, int seq_len, int row_stride, float* scratch, float p[3]) const {
            const int H = hidden, G = 3 * hidden;
            float* gi = scratch;        // [3H]
            float* gh = gi + G;         // [3H]
            float* h = gh + G;          // [layers][H]
            const int n_layers = static_cast<int>(layers.size());
            std::memset(h, 0, sizeof(float) * H * n_layers);
            for (int t = 0; t < seq_len; ++t) {
                const float* x = window + static_cast<size_t>(t) * row_stride;
                for (int l = 0; l < n_layers; ++l) {
                    const Layer& L = layers[l];
                    float* hl = h + l * H;
                    affine(L.w_ih_t.data(), L.b_ih.data(), x, L.in, G, gi);
                    affine(L.w_hh_t.data(), L.b_hh.data(), hl, H, G, gh);
                    for (int j = 0; j < H; ++j) {
                        const float r = sigmoid(gi[j] + gh[j]);
                        const float z = sigmoid(gi[H + j] + gh[H + j]);
                        const float n = vtanh(gi[2 * H + j] + r * gh[2 * H + j]);
                        hl[j] = (1.0f - z) * n + z * hl[j];
                    }
                    x = hl;
                }
            }
            // Head over [top-layer h_last, x_last (first n_inputs features)]
            const float* top = h + (n_layers - 1) * H;
            const float* x_last = window + static_cast<size_t>(seq_len - 1) * row_stride;
            float logits[3] = {head_b[0], head_b[1], head_b[2]};
            for (int c = 0; c < H; ++c)
                for (int k = 0; k < 3; ++k) logits[k] += top[c] * head_w_t[static_cast<size_t>(c) * 3 + k];
            for (int c = 0; c < n_inputs; ++c)
                for (int k = 0; k < 3; ++k) logits[k] += x_last[c] * head_w_t[static_cast<size_t>(H + c) * 3 + k];
            float mx = -INFINITY;
            for (int k = 0; k < 3; ++k) {
                logits[k] = logits[k] * inv_t + cal_b[k];
                mx = std::max(mx, logits[k]);
            }
            float sum = 0.0f;
            for (int k = 0; k < 3; ++k) sum += (p[k] = std::exp(logits[k] - mx));
            for (int k = 0; k < 3; ++k) p[k] /= sum;
        }
    };

    static int32_t read_i32(std::ifstream& in) {
        int32_t v = 0;
        in.read(reinterpret_cast<char*>(&v), sizeof(v));
        return v;
    }

    static std::vector<float> read_f32(std::ifstream& in, size_t n) {
        std::vector<float> v(n);
        in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(float)));
        return v;
    }

    static std::vector<float> transpose(const std::vector<float>& m, int rows, int cols) {
        std::vector<float> t(m.size());
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) t[static_cast<size_t>(c) * rows + r] = m[static_cast<size_t>(r) * cols + c];
        return t;
    }

    Combine combine_ = kSingle;
    std::vector<Net> nets_;
    int input_dim_ = 0;
    std::vector<float> scratch_;
};

#endif
