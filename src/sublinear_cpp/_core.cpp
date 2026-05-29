#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <set>
#include <fstream>
#include <stdexcept>
#include <climits>
#include <limits>


uint32_t fnv1a_hash(const std::string& data, uint32_t seed) {
    uint32_t hash = 2166136261U ^ seed;
    for (char c : data) {
        hash ^= static_cast<uint8_t>(c);
        hash *= 16777619U;
    }
    return hash;
}

class CountMinSketch {
protected:
    uint32_t width;
    uint32_t depth;
    std::vector<std::vector<uint32_t>> table;

public:
    CountMinSketch(uint32_t w, uint32_t d) 
        : width(w), depth(d), table(d, std::vector<uint32_t>(w, 0)) {
        if (w == 0 || d == 0) throw std::invalid_argument("width and depth must be > 0");
    }

    virtual ~CountMinSketch() = default;

    // Core Conservative Update Logic
    virtual uint32_t add(const std::string& word) {
        std::vector<uint32_t> indices(depth);
        uint32_t min_value = std::numeric_limits<uint32_t>::max();

        for (uint32_t i = 0; i < depth; ++i) {
            indices[i] = fnv1a_hash(word, i) % width;
            if (table[i][indices[i]] < min_value) {
                min_value = table[i][indices[i]];
            }
        }

        for (uint32_t i = 0; i < depth; ++i) {
            if (table[i][indices[i]] == min_value) {
                if (table[i][indices[i]] != std::numeric_limits<uint32_t>::max())
                    table[i][indices[i]]++;
            }
        }

        return estimate(word);
    }

    uint32_t estimate(const std::string& word) const {
        uint32_t min_value = std::numeric_limits<uint32_t>::max();
        for (uint32_t i = 0; i < depth; ++i) {
            uint32_t index = fnv1a_hash(word, i) % width;
            if (table[i][index] < min_value) {
                min_value = table[i][index];
            }
        }
        return min_value;
    }

    void save(const std::string& filepath) const {
        std::ofstream out(filepath, std::ios::binary);
        if (!out) throw std::runtime_error("Cannot open file for writing");
        uint32_t version = 1;
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        out.write(reinterpret_cast<const char*>(&width), sizeof(width));
        out.write(reinterpret_cast<const char*>(&depth), sizeof(depth));
        for (const auto& row : table) {
            out.write(reinterpret_cast<const char*>(row.data()), width * sizeof(uint32_t));
        }
        if (!out) throw std::runtime_error("Write error while saving CountMinSketch");
    }

    void load(const std::string& filepath) {
        std::ifstream in(filepath, std::ios::binary);
        if (!in) throw std::runtime_error("Cannot open file for reading");
        uint32_t version = 0;
        in.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (!in) throw std::runtime_error("Read error while loading CountMinSketch (version)");
        if (version != 1) throw std::runtime_error("Unsupported CountMinSketch file version");
        in.read(reinterpret_cast<char*>(&width), sizeof(width));
        in.read(reinterpret_cast<char*>(&depth), sizeof(depth));
        if (!in) throw std::runtime_error("Read error while loading CountMinSketch (dims)");
        if (width == 0 || depth == 0) throw std::runtime_error("Invalid width/depth in file");
        table.assign(depth, std::vector<uint32_t>(width, 0));
        for (auto& row : table) {
            in.read(reinterpret_cast<char*>(row.data()), width * sizeof(uint32_t));
            if (!in) throw std::runtime_error("Read error while loading CountMinSketch (table)");
        }
    }
};

class TopKCountMinSketch : public CountMinSketch {
private:
    uint32_t k_size;
    std::unordered_map<std::string, uint32_t> top_k_map;
    std::set<std::pair<uint32_t, std::string>> top_k_set;

public:
    TopKCountMinSketch(uint32_t w, uint32_t d, uint32_t k) 
        : CountMinSketch(w, d), k_size(k) {}
    uint32_t add(const std::string& word) override {
        uint32_t new_estimate = CountMinSketch::add(word);

        auto it = top_k_map.find(word);
        if (it != top_k_map.end()) {
            uint32_t old_est = it->second;
            top_k_set.erase({old_est, word});
            top_k_set.insert({new_estimate, word});
            it->second = new_estimate;
        } else {
            if (top_k_map.size() < k_size) {
                top_k_set.insert({new_estimate, word});
                top_k_map[word] = new_estimate;
            } else if (new_estimate > top_k_set.begin()->first) {
                auto min_el = top_k_set.begin();
                top_k_map.erase(min_el->second);
                top_k_set.erase(min_el);

                top_k_set.insert({new_estimate, word});
                top_k_map[word] = new_estimate;
            }
        }

        return new_estimate;
    }

    std::vector<uint32_t> get_top_k_frequencies() const {
        std::vector<uint32_t> freqs;
        freqs.reserve(top_k_set.size());
        for (auto it = top_k_set.rbegin(); it != top_k_set.rend(); ++it) {
            freqs.push_back(it->first);
        }
        return freqs;
    }

    std::vector<std::pair<std::string, uint32_t>> get_top_k_items() const {
        std::vector<std::pair<std::string, uint32_t>> items;
        items.reserve(top_k_set.size());
        for (auto it = top_k_set.rbegin(); it != top_k_set.rend(); ++it) {
            items.emplace_back(it->second, it->first);
        }
        return items;
    }
};


namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "High-performance C++ sublinear data structures";

    // Standard raw CMS
    py::class_<CountMinSketch>(m, "CountMinSketch")
        .def(py::init<uint32_t, uint32_t>(), py::arg("width"), py::arg("depth"))
        .def("add", &CountMinSketch::add)
        .def("estimate", &CountMinSketch::estimate)
        .def("save", &CountMinSketch::save)
        .def("load", &CountMinSketch::load);

    // Extended Top-K CMS
    py::class_<TopKCountMinSketch, CountMinSketch>(m, "TopKCountMinSketch")
        .def(py::init<uint32_t, uint32_t, uint32_t>(), py::arg("width"), py::arg("depth"), py::arg("k_size"))
        .def("add", &TopKCountMinSketch::add)
        .def("get_top_k_frequencies", &TopKCountMinSketch::get_top_k_frequencies)
        .def("get_top_k_items", &TopKCountMinSketch::get_top_k_items);
}