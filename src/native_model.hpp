#ifndef NATIVE_MODEL_HPP
#define NATIVE_MODEL_HPP

// Dependency-free inference for the models exported by python/native_model.py.
// Replaces libtorch: the weights are read once into flat arrays and the GRU runs as
// plain vectorisable loops, with no interpreter, dispatcher or tensor allocation.
//
// File layout (little endian, version 2):
//   "QGRU" | int32 version | int32 combine | int32 n_groups
//   per group: int32 n_nets, then per net (one per random start of the same model):
//            int32 n_inputs, hidden, layers
//            per layer: weight_ih [3H x in], weight_hh [3H x H], bias_ih [3H], bias_hh [3H]
//                       (PyTorch gate order r, z, n)
//            head weight [3 x (H + n_inputs)], head bias [3]
//            calibration: float inv_temperature, float bias[3]
// Each net reads the first n_inputs features of every window row and outputs
//   logits = head([h_last, x_last]) * inv_temperature + bias.
// A group's probabilities are the mean over its nets' softmax outputs.
// combine: 0 single group | 1 mean of the two groups' probabilities |
//          2 group0 probabilities, HOLD when group1 disagrees on direction |
//          3 same, but only group0 SELL calls need agreement
//
// Online learning: predict() can return each net's head input z = [h_last, x_last]; once the
// bar's label is known, learn() takes one SGD step per net on its own cross-entropy, on the
// head and calibration only (the GRU stays fixed), with an L2 pull back to the trained values
// so the model cannot drift far on noise. save()/adopt() persist the adapted values.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

class NativeModel {
public:
    enum Combine : int32_t { kSingle = 0, kMean = 1, kVeto = 2, kVetoShort = 3 };

    static NativeModel load(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("cannot open model " + path);
        NativeModel m;
        m.bytes_.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        Reader in{m.bytes_, 0, path};
        if (m.bytes_.size() < 4 || std::memcmp(m.bytes_.data(), "QGRU", 4) != 0) throw std::runtime_error(path + " is not a QGRU model file");
        in.pos = 4;
        const int32_t version = in.i32();
        if (version != 2)
            throw std::runtime_error(path + ": model format version " + std::to_string(version) +
                                     " is not supported; re-export it with python/train_and_export.py");
        m.combine_ = static_cast<Combine>(in.i32());
        const int32_t n_groups = in.i32();
        const int32_t expected_groups = m.combine_ == kSingle ? 1 : 2;
        if (m.combine_ < kSingle || m.combine_ > kVetoShort || n_groups != expected_groups)
            throw std::runtime_error(path + ": bad combine mode / group count");
        for (int g = 0; g < n_groups; ++g) {
            const int32_t n_nets = in.i32();
            if (n_nets < 1 || n_nets > 64) throw std::runtime_error(path + ": implausible number of nets in a group");
            m.groups_.emplace_back();
            for (int k = 0; k < n_nets; ++k) m.groups_.back().push_back(Net::read(in));
        }
        if (in.pos != m.bytes_.size()) throw std::runtime_error(path + ": unexpected trailing data");
        size_t scratch = 0;
        for (const auto& group : m.groups_) {
            for (const auto& n : group) {
                m.input_dim_ = std::max(m.input_dim_, n.n_inputs);
                scratch = std::max(scratch, static_cast<size_t>(n.hidden) * (6 + n.layers.size())); // gi, gh, h per layer
            }
        }
        m.scratch_.assign(scratch, 0.0f);
        return m;
    }

    int input_dim() const { return input_dim_; }

    // Floats predict() writes to z: every net's head input, in file order
    size_t z_size() const {
        size_t n = 0;
        for (const auto& g : groups_)
            for (const auto& net : g) n += static_cast<size_t>(net.hidden + net.n_inputs);
        return n;
    }

    // FNV-1a of the file as loaded; ties a saved online-learning state to its base model
    uint64_t fingerprint() const {
        uint64_t h = 1469598103934665603ull;
        for (unsigned char c : bytes_) h = (h ^ c) * 1099511628211ull;
        return h;
    }

    // window: seq_len rows of row_stride floats (oldest first). Writes SELL/HOLD/BUY probabilities,
    // and each net's head input into z (z_size() floats) when z is given.
    void predict(const float* window, int seq_len, int row_stride, float probs[3], float* z = nullptr) {
        float p0[3], p1[3];
        group_probs(groups_[0], window, seq_len, row_stride, p0, z);
        if (combine_ == kSingle) {
            std::memcpy(probs, p0, sizeof(p0));
            return;
        }
        group_probs(groups_[1], window, seq_len, row_stride, p1, z ? z + group_z(groups_[0]) : nullptr);
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

    // One SGD step per net on its cross-entropy for `label` (0 SELL, 1 HOLD, 2 BUY), given the
    // z that predict() returned for that bar. anchor = L2 pull towards the trained values.
    void learn(const float* z, int label, float lr, float anchor) {
        for (auto& g : groups_)
            for (auto& net : g) {
                net.learn(z, label, lr, anchor);
                z += net.hidden + net.n_inputs;
            }
    }

    // The model with its current (possibly adapted) head and calibration, same format as loaded
    void save(const std::string& path) const {
        std::string out = bytes_;
        for (const auto& g : groups_)
            for (const auto& net : g) net.write_head(out);
        const std::string tmp = path + ".tmp";
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            f.write(out.data(), static_cast<std::streamsize>(out.size()));
            if (!f) throw std::runtime_error("cannot write " + tmp);
        }
        if (std::rename(tmp.c_str(), path.c_str()) != 0) throw std::runtime_error("cannot replace " + path);
    }

    // Take the adapted head/calibration values of `state` (same architecture), keep our anchors
    bool adopt(const NativeModel& state) {
        if (state.groups_.size() != groups_.size() || state.combine_ != combine_) return false;
        for (size_t g = 0; g < groups_.size(); ++g) {
            if (state.groups_[g].size() != groups_[g].size()) return false;
            for (size_t k = 0; k < groups_[g].size(); ++k)
                if (!groups_[g][k].same_shape(state.groups_[g][k])) return false;
        }
        for (size_t g = 0; g < groups_.size(); ++g)
            for (size_t k = 0; k < groups_[g].size(); ++k) groups_[g][k].take_head(state.groups_[g][k]);
        return true;
    }

private:
    struct Reader {
        const std::string& b;
        size_t pos;
        const std::string& path;
        int32_t i32() {
            int32_t v = 0;
            need(sizeof(v));
            std::memcpy(&v, b.data() + pos, sizeof(v));
            pos += sizeof(v);
            return v;
        }
        std::vector<float> f32(size_t n) {
            std::vector<float> v(n);
            need(n * sizeof(float));
            std::memcpy(v.data(), b.data() + pos, n * sizeof(float));
            pos += n * sizeof(float);
            return v;
        }
        void need(size_t n) const {
            if (pos + n > b.size()) throw std::runtime_error(path + ": truncated model file");
        }
    };

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
        // Trained values (anchors for online learning) and where the head sits in the file
        std::vector<float> head_w0_t;
        float head_b0[3] = {0, 0, 0}, inv_t0 = 1.0f, cal_b0[3] = {0, 0, 0};
        size_t head_offset = 0;

        static Net read(Reader& in) {
            Net n;
            n.n_inputs = in.i32();
            n.hidden = in.i32();
            const int n_layers = in.i32();
            if (n.n_inputs <= 0 || n.hidden <= 0 || n_layers <= 0 || n.n_inputs > 4096 || n.hidden > 4096 || n_layers > 16)
                throw std::runtime_error(in.path + ": implausible network shape");
            const int g = 3 * n.hidden;
            for (int l = 0; l < n_layers; ++l) {
                Layer L;
                L.in = l == 0 ? n.n_inputs : n.hidden;
                L.w_ih_t = transpose(in.f32(static_cast<size_t>(g) * L.in), g, L.in);
                L.w_hh_t = transpose(in.f32(static_cast<size_t>(g) * n.hidden), g, n.hidden);
                L.b_ih = in.f32(g);
                L.b_hh = in.f32(g);
                n.layers.push_back(std::move(L));
            }
            n.head_offset = in.pos;
            n.head_w_t = transpose(in.f32(static_cast<size_t>(3) * (n.hidden + n.n_inputs)), 3, n.hidden + n.n_inputs);
            const auto hb = in.f32(3);
            std::memcpy(n.head_b, hb.data(), sizeof(n.head_b));
            n.inv_t = in.f32(1)[0];
            const auto cb = in.f32(3);
            std::memcpy(n.cal_b, cb.data(), sizeof(n.cal_b));
            n.head_w0_t = n.head_w_t;
            std::memcpy(n.head_b0, n.head_b, sizeof(n.head_b));
            n.inv_t0 = n.inv_t;
            std::memcpy(n.cal_b0, n.cal_b, sizeof(n.cal_b));
            return n;
        }

        bool same_shape(const Net& o) const {
            return n_inputs == o.n_inputs && hidden == o.hidden && layers.size() == o.layers.size();
        }

        void take_head(const Net& o) {
            head_w_t = o.head_w_t;
            std::memcpy(head_b, o.head_b, sizeof(head_b));
            inv_t = o.inv_t;
            std::memcpy(cal_b, o.cal_b, sizeof(cal_b));
        }

        void write_head(std::string& out) const {
            const int D = hidden + n_inputs;
            std::vector<float> w(static_cast<size_t>(3) * D); // back to PyTorch's row-major [3][D]
            for (int c = 0; c < D; ++c)
                for (int k = 0; k < 3; ++k) w[static_cast<size_t>(k) * D + c] = head_w_t[static_cast<size_t>(c) * 3 + k];
            size_t pos = head_offset;
            auto put = [&](const float* v, size_t n) {
                std::memcpy(out.data() + pos, v, n * sizeof(float));
                pos += n * sizeof(float);
            };
            put(w.data(), w.size());
            put(head_b, 3);
            put(&inv_t, 1);
            put(cal_b, 3);
        }

        void learn(const float* z, int label, float lr, float anchor) {
            const int D = hidden + n_inputs;
            float raw[3] = {head_b[0], head_b[1], head_b[2]};
            for (int c = 0; c < D; ++c)
                for (int k = 0; k < 3; ++k) raw[k] += z[c] * head_w_t[static_cast<size_t>(c) * 3 + k];
            float logit[3], p[3], mx = -INFINITY, sum = 0.0f;
            for (int k = 0; k < 3; ++k) mx = std::max(mx, logit[k] = raw[k] * inv_t + cal_b[k]);
            for (int k = 0; k < 3; ++k) sum += (p[k] = std::exp(logit[k] - mx));
            float g[3];
            for (int k = 0; k < 3; ++k) g[k] = p[k] / sum - (k == label ? 1.0f : 0.0f); // dCE/dlogit
            float g_inv_t = 0.0f;
            for (int k = 0; k < 3; ++k) g_inv_t += g[k] * raw[k];
            for (int c = 0; c < D; ++c)
                for (int k = 0; k < 3; ++k) {
                    const size_t i = static_cast<size_t>(c) * 3 + k;
                    head_w_t[i] -= lr * (inv_t * g[k] * z[c] + anchor * (head_w_t[i] - head_w0_t[i]));
                }
            for (int k = 0; k < 3; ++k) {
                head_b[k] -= lr * (inv_t * g[k] + anchor * (head_b[k] - head_b0[k]));
                cal_b[k] -= lr * (g[k] + anchor * (cal_b[k] - cal_b0[k]));
            }
            inv_t = std::clamp(inv_t - lr * (g_inv_t + anchor * (inv_t - inv_t0)), 0.2f, 5.0f);
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

        void probs(const float* window, int seq_len, int row_stride, float* scratch, float p[3], float* z = nullptr) const {
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
            if (z) {
                std::memcpy(z, top, sizeof(float) * H);
                std::memcpy(z + H, x_last, sizeof(float) * n_inputs);
            }
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

    static std::vector<float> transpose(const std::vector<float>& m, int rows, int cols) {
        std::vector<float> t(m.size());
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) t[static_cast<size_t>(c) * rows + r] = m[static_cast<size_t>(r) * cols + c];
        return t;
    }

    static size_t group_z(const std::vector<Net>& group) {
        size_t n = 0;
        for (const Net& net : group) n += static_cast<size_t>(net.hidden + net.n_inputs);
        return n;
    }

    void group_probs(const std::vector<Net>& group, const float* window, int seq_len, int row_stride, float p[3], float* z) {
        p[0] = p[1] = p[2] = 0.0f;
        for (const Net& net : group) {
            float q[3];
            net.probs(window, seq_len, row_stride, scratch_.data(), q, z);
            if (z) z += net.hidden + net.n_inputs;
            for (int c = 0; c < 3; ++c) p[c] += q[c];
        }
        const float inv = 1.0f / static_cast<float>(group.size());
        for (int c = 0; c < 3; ++c) p[c] *= inv;
    }

    Combine combine_ = kSingle;
    std::vector<std::vector<Net>> groups_;
    std::string bytes_; // the file as loaded (save() patches the head values into a copy)
    int input_dim_ = 0;
    std::vector<float> scratch_;
};

#endif
