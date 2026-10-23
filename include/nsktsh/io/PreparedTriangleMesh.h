#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace nsktsh::io {

struct PreparedMeshNode { double x{}, y{}; };
struct PreparedMeshTriangle { std::array<std::uint32_t, 3> nodes{}; };
struct PreparedMeshBoundaryEdge { std::array<std::uint32_t, 2> nodes{}; int tag{}; };

struct PreparedTriangleMesh {
    // All returned indices are zero-based. Boundary line orientation is retained.
    std::vector<PreparedMeshNode> nodes;
    std::vector<PreparedMeshTriangle> triangles;
    std::vector<PreparedMeshBoundaryEdge> boundary_edges;
    double min_area{};
    double area{};
};

struct PreparedTriangleMeshLimits {
    std::uint64_t maximum_nodes{2'000'000U};
    std::uint64_t maximum_triangles{2'000'000U};
    std::uintmax_t maximum_bytes{256ULL * 1024ULL * 1024ULL};
};

struct PreparedMeshRectangle { double xllcorner{}, yllcorner{}, width{}, height{}; };

struct PreparedTriangleMeshReadResult {
    std::optional<PreparedTriangleMesh> mesh;
    std::string code;
    std::string message;
    [[nodiscard]] bool Succeeded() const noexcept {
        return mesh.has_value() && code == "prepared_mesh_read";
    }
};

namespace prepared_mesh_detail {

struct Failure { const char* code; const char* message; };

inline void Require(bool condition, const char* code, const char* message) {
    if (!condition) throw Failure{code, message};
}

class Cancellation {
public:
    Cancellation(std::stop_token token, const std::function<bool()>& callback)
        : token_(token), callback_(callback) {}
    void Check() const {
        if (token_.stop_requested() || (callback_ && callback_()))
            throw Failure{"prepared_mesh_read_cancelled", "Prepared mesh reading was cancelled."};
    }
    void Tick() {
        if ((++ticks_ & 4095U) == 0U) Check();
    }
private:
    std::stop_token token_;
    const std::function<bool()>& callback_;
    std::uint32_t ticks_{};
};

class Tokens {
public:
    Tokens(std::string_view input, Cancellation& cancellation)
        : input_(input), cancellation_(cancellation) {}
    bool Next(std::string_view& token) {
        SkipSpace();
        if (input_.empty()) return false;
        std::size_t length = 0U;
        while (length < input_.size() && !Space(input_[length])) {
            if (++length > 256U) return false;
            cancellation_.Tick();
        }
        token = input_.substr(0U, length);
        input_.remove_prefix(length);
        return true;
    }
    bool Expect(std::string_view expected) {
        std::string_view token;
        return Next(token) && token == expected;
    }
    bool Unsigned(std::uint64_t& value) {
        std::string_view token;
        if (!Next(token) || token.front() == '-' || token.front() == '+') return false;
        const auto result = std::from_chars(token.data(), token.data() + token.size(), value);
        return result.ec == std::errc{} && result.ptr == token.data() + token.size();
    }
    bool Integer(std::uint64_t expected) {
        std::uint64_t value{};
        return Unsigned(value) && value == expected;
    }
    bool Number(double& value) {
        std::string_view token;
        if (!Next(token)) return false;
        const auto result = std::from_chars(token.data(), token.data() + token.size(), value,
            std::chars_format::general);
        return result.ec == std::errc{} && result.ptr == token.data() + token.size() && std::isfinite(value);
    }
    bool Finished() { SkipSpace(); return input_.empty(); }
private:
    static bool Space(char c) noexcept {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    }
    void SkipSpace() {
        while (!input_.empty() && Space(input_.front())) {
            input_.remove_prefix(1U);
            cancellation_.Tick();
        }
    }
    std::string_view input_;
    Cancellation& cancellation_;
};

inline std::uint64_t EdgeKey(std::uint32_t a, std::uint32_t b) noexcept {
    if (a > b) std::swap(a, b);
    return (static_cast<std::uint64_t>(a) << 32U) | b;
}

struct Edge {
    std::uint64_t key;
    std::uint32_t triangle;
    bool ascending;
};

class Components {
public:
    explicit Components(std::size_t count) : parents_(count), ranks_(count), count_(count) {
        std::iota(parents_.begin(), parents_.end(), 0U);
    }
    void Join(std::uint32_t a, std::uint32_t b) {
        a = Root(a); b = Root(b);
        if (a == b) return;
        if (ranks_[a] < ranks_[b]) std::swap(a, b);
        parents_[b] = a;
        if (ranks_[a] == ranks_[b]) ++ranks_[a];
        --count_;
    }
    [[nodiscard]] std::size_t Count() const noexcept { return count_; }
private:
    std::uint32_t Root(std::uint32_t node) {
        while (parents_[node] != node) {
            parents_[node] = parents_[parents_[node]];
            node = parents_[node];
        }
        return node;
    }
    std::vector<std::uint32_t> parents_;
    std::vector<std::uint8_t> ranks_;
    std::size_t count_;
};

inline PreparedTriangleMesh Read(std::string_view bytes, PreparedMeshRectangle rectangle,
    PreparedTriangleMeshLimits limits, Cancellation& cancellation) {
    cancellation.Check();
    Require(limits.maximum_bytes > 0U && limits.maximum_nodes > 0U && limits.maximum_triangles > 0U,
        "prepared_mesh_limit_invalid", "Prepared mesh limits must be positive.");
    Require(bytes.size() <= limits.maximum_bytes, "prepared_mesh_byte_limit_exceeded",
        "Prepared mesh exceeds its byte limit.");
    const double right = rectangle.xllcorner + rectangle.width;
    const double top = rectangle.yllcorner + rectangle.height;
    const double width = right - rectangle.xllcorner;
    const double height = top - rectangle.yllcorner;
    Require(std::isfinite(rectangle.xllcorner) && std::isfinite(rectangle.yllcorner) &&
        std::isfinite(rectangle.width) && std::isfinite(rectangle.height) &&
        rectangle.width > 0.0 && rectangle.height > 0.0 && std::isfinite(right) &&
        std::isfinite(top) && std::isfinite(width) && std::isfinite(height) && width > 0.0 && height > 0.0 &&
        std::isfinite(width * height) && width * height > 0.0,
        "prepared_mesh_rectangle_invalid", "Prepared mesh rectangle must have representable positive extents and area.");
    Tokens tokens(bytes, cancellation);
    const auto syntax = [](bool valid) {
        Require(valid, "prepared_mesh_format_invalid", "Expected complete canonical Gmsh 2.2 ASCII prepared mesh.");
    };
    syntax(tokens.Expect("$MeshFormat") && tokens.Expect("2.2") && tokens.Integer(0U) && tokens.Integer(8U) &&
        tokens.Expect("$EndMeshFormat") && tokens.Expect("$PhysicalNames") && tokens.Integer(5U));
    constexpr const char* names[] = {"\"bottom\"", "\"right\"", "\"top\"", "\"left\"", "\"domain\""};
    for (std::uint64_t i = 0U; i < 5U; ++i)
        syntax(tokens.Integer(i < 4U ? 1U : 2U) && tokens.Integer(i + 1U) && tokens.Expect(names[i]));
    syntax(tokens.Expect("$EndPhysicalNames") && tokens.Expect("$Nodes"));
    std::uint64_t node_count{};
    syntax(tokens.Unsigned(node_count));
    Require(node_count >= 4U && node_count <= limits.maximum_nodes &&
        node_count <= std::numeric_limits<std::uint32_t>::max() && node_count <= bytes.size() / 8U &&
        node_count <= std::vector<PreparedMeshNode>().max_size(),
        "prepared_mesh_node_limit_exceeded", "Prepared mesh node count is invalid or exceeds its limits.");
    PreparedTriangleMesh mesh;
    mesh.nodes.reserve(static_cast<std::size_t>(node_count));
    for (std::uint64_t i = 0U; i < node_count; ++i) {
        if ((i & 4095U) == 0U) cancellation.Check();
        PreparedMeshNode node;
        double z{};
        syntax(tokens.Integer(i + 1U) && tokens.Number(node.x) && tokens.Number(node.y) && tokens.Number(z));
        Require(z == 0.0 && node.x >= rectangle.xllcorner && node.x <= right &&
            node.y >= rectangle.yllcorner && node.y <= top,
            "prepared_mesh_node_invalid", "Prepared mesh nodes must have zero Z and lie inside the rectangle.");
        mesh.nodes.push_back(node);
    }
    // Sort indices, retaining producer order in the returned geometry.
    std::vector<std::uint32_t> order(mesh.nodes.size());
    std::iota(order.begin(), order.end(), 0U);
    std::sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
        cancellation.Tick();
        const auto& p = mesh.nodes[a]; const auto& q = mesh.nodes[b];
        return p.x < q.x || (p.x == q.x && p.y < q.y);
    });
    for (std::size_t i = 1U; i < order.size(); ++i) {
        cancellation.Tick();
        const auto& p = mesh.nodes[order[i - 1U]]; const auto& q = mesh.nodes[order[i]];
        Require(p.x != q.x || p.y != q.y, "prepared_mesh_node_duplicate", "Prepared mesh has duplicate XY nodes.");
    }
    syntax(tokens.Expect("$EndNodes") && tokens.Expect("$Elements"));
    std::uint64_t element_count{};
    syntax(tokens.Unsigned(element_count));
    const auto triangle_limit = std::min<std::uint64_t>(limits.maximum_triangles,
        std::numeric_limits<std::uint32_t>::max());
    Require(element_count >= 6U && element_count <= triangle_limit + node_count &&
        element_count <= bytes.size() / 12U,
        "prepared_mesh_element_limit_exceeded", "Prepared mesh element count is invalid or exceeds its limits.");
    const auto triangle_reserve = static_cast<std::size_t>(std::min(element_count, triangle_limit));
    Require(triangle_reserve <= mesh.triangles.max_size() && triangle_reserve <= std::vector<Edge>().max_size() / 3U,
        "prepared_mesh_element_limit_exceeded", "Prepared mesh elements exceed addressable storage.");
    mesh.triangles.reserve(triangle_reserve);
    std::vector<std::uint8_t> used(mesh.nodes.size());
    long double area_sum = 0.0L, area_correction = 0.0L;
    mesh.min_area = std::numeric_limits<double>::max();
    for (std::uint64_t i = 0U; i < element_count; ++i) {
        if ((i & 4095U) == 0U) cancellation.Check();
        std::uint64_t type{}, tag{}, entity{}, a{}, b{}, c{};
        syntax(tokens.Integer(i + 1U) && tokens.Unsigned(type) && tokens.Integer(2U) &&
            tokens.Unsigned(tag) && tokens.Unsigned(entity) && tokens.Unsigned(a) && tokens.Unsigned(b));
        syntax((type == 1U || type == 2U) && tag == entity && a > 0U && a <= node_count && b > 0U && b <= node_count);
        --a; --b;
        Require(a != b, "prepared_mesh_element_invalid", "Prepared mesh element repeats a node.");
        if (type == 1U) {
            syntax(mesh.triangles.empty() && tag >= 1U && tag <= 4U);
            Require(mesh.boundary_edges.size() < node_count, "prepared_mesh_boundary_invalid",
                "Prepared mesh has too many boundary edges for a simple rectangle.");
            mesh.boundary_edges.push_back({{static_cast<std::uint32_t>(a), static_cast<std::uint32_t>(b)}, static_cast<int>(tag)});
            continue;
        }
        syntax(tag == 5U && tokens.Unsigned(c) && c > 0U && c <= node_count);
        --c;
        Require(c != a && c != b, "prepared_mesh_element_invalid", "Prepared mesh triangle repeats a node.");
        Require(mesh.triangles.size() < triangle_limit, "prepared_mesh_triangle_limit_exceeded",
            "Prepared mesh exceeds its triangle limit.");
        const auto& p = mesh.nodes[static_cast<std::size_t>(a)];
        const auto& q = mesh.nodes[static_cast<std::size_t>(b)];
        const auto& r = mesh.nodes[static_cast<std::size_t>(c)];
        const long double dx1 = static_cast<long double>(q.x) - p.x;
        const long double dy1 = static_cast<long double>(q.y) - p.y;
        const long double dx2 = static_cast<long double>(r.x) - p.x;
        const long double dy2 = static_cast<long double>(r.y) - p.y;
        const long double product1 = dx1 * dy2, product2 = dy1 * dx2;
        const long double determinant = product1 - product2;
        // Reject uncertain orientation instead of admitting a rounded degeneracy.
        const long double uncertainty = 16.0L * std::numeric_limits<long double>::epsilon() *
            (std::abs(product1) + std::abs(product2));
        const double area = 0.5 * ((q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x));
        Require(std::isfinite(determinant) && determinant > uncertainty && std::isfinite(area) && area > 0.0,
            "prepared_mesh_triangle_geometry_invalid", "Prepared mesh triangles must be finite, strictly CCW and numerically nondegenerate.");
        const long double term = determinant * 0.5L - area_correction;
        const long double next_sum = area_sum + term;
        area_correction = (next_sum - area_sum) - term;
        area_sum = next_sum;
        mesh.min_area = std::min(mesh.min_area, area);
        mesh.triangles.push_back({{static_cast<std::uint32_t>(a), static_cast<std::uint32_t>(b), static_cast<std::uint32_t>(c)}});
        used[static_cast<std::size_t>(a)] = used[static_cast<std::size_t>(b)] = used[static_cast<std::size_t>(c)] = 1U;
    }
    syntax(tokens.Expect("$EndElements") && tokens.Finished());
    Require(mesh.triangles.size() >= 2U && mesh.boundary_edges.size() >= 4U,
        "prepared_mesh_topology_invalid", "Prepared mesh needs triangles and a closed rectangular boundary.");
    for (const auto value : used) {
        cancellation.Tick();
        Require(value != 0U, "prepared_mesh_node_unused", "Every prepared mesh node must belong to a triangle.");
    }
    const long double expected_area = static_cast<long double>(width) * height;
    Require(std::isfinite(area_sum) && std::abs(area_sum - expected_area) <=
        128.0L * std::numeric_limits<double>::epsilon() * expected_area,
        "prepared_mesh_area_invalid", "Prepared mesh area does not cover its rectangle exactly once.");
    mesh.area = static_cast<double>(area_sum);
    std::vector<Edge> edges;
    edges.reserve(mesh.triangles.size() * 3U);
    for (std::size_t i = 0U; i < mesh.triangles.size(); ++i) {
        cancellation.Tick();
        const auto& triangle = mesh.triangles[i];
        for (std::size_t j = 0U; j < 3U; ++j) {
            const auto a = triangle.nodes[j], b = triangle.nodes[(j + 1U) % 3U];
            edges.push_back({EdgeKey(a, b), static_cast<std::uint32_t>(i), a < b});
        }
    }
    std::sort(edges.begin(), edges.end(), [&](const Edge& a, const Edge& b) {
        cancellation.Tick(); return a.key < b.key;
    });
    order.resize(mesh.boundary_edges.size());
    std::iota(order.begin(), order.end(), 0U);
    const auto boundary_key = [&](std::uint32_t index) {
        const auto& edge = mesh.boundary_edges[index]; return EdgeKey(edge.nodes[0], edge.nodes[1]);
    };
    std::sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
        cancellation.Tick(); return boundary_key(a) < boundary_key(b);
    });
    for (std::size_t i = 1U; i < order.size(); ++i) {
        cancellation.Tick();
        Require(boundary_key(order[i - 1U]) != boundary_key(order[i]), "prepared_mesh_boundary_invalid",
            "Prepared mesh declares a boundary edge more than once.");
    }
    Components components(mesh.triangles.size());
    std::size_t unique_edges = 0U, unmatched = 0U;
    for (std::size_t i = 0U; i < edges.size();) {
        cancellation.Tick();
        std::size_t end = i + 1U;
        while (end < edges.size() && edges[end].key == edges[i].key) { cancellation.Tick(); ++end; }
        ++unique_edges;
        Require(end - i <= 2U, "prepared_mesh_topology_invalid", "Prepared mesh has a non-manifold edge.");
        if (end - i == 2U) {
            // This also rejects duplicate CCW triangles, including cyclic permutations.
            Require(edges[i].ascending != edges[i + 1U].ascending, "prepared_mesh_topology_invalid",
                "Prepared mesh triangles share an edge with the same orientation.");
            components.Join(edges[i].triangle, edges[i + 1U].triangle);
        } else {
            ++unmatched;
            const auto found = std::lower_bound(order.begin(), order.end(), edges[i].key,
                [&](std::uint32_t index, std::uint64_t key) { return boundary_key(index) < key; });
            Require(found != order.end() && boundary_key(*found) == edges[i].key,
                "prepared_mesh_boundary_invalid", "Prepared mesh has an undeclared or hanging boundary edge.");
            const auto& boundary = mesh.boundary_edges[*found];
            auto a = static_cast<std::uint32_t>(edges[i].key >> 32U);
            auto b = static_cast<std::uint32_t>(edges[i].key & 0xffffffffULL);
            if (!edges[i].ascending) std::swap(a, b);
            const auto& p = mesh.nodes[a]; const auto& q = mesh.nodes[b];
            int side = 0;
            if (p.y == rectangle.yllcorner && q.y == rectangle.yllcorner && q.x > p.x) side = 1;
            else if (p.x == right && q.x == right && q.y > p.y) side = 2;
            else if (p.y == top && q.y == top && q.x < p.x) side = 3;
            else if (p.x == rectangle.xllcorner && q.x == rectangle.xllcorner && q.y < p.y) side = 4;
            Require(side != 0 && boundary.tag == side, "prepared_mesh_boundary_invalid",
                "Prepared mesh boundary tag or oriented triangle edge disagrees with its rectangle side.");
        }
        i = end;
    }
    Require(unmatched == mesh.boundary_edges.size(), "prepared_mesh_boundary_invalid",
        "Declared prepared mesh boundary must equal all and only unmatched triangle edges.");
    Require(components.Count() == 1U && mesh.nodes.size() + mesh.triangles.size() == unique_edges + 1U,
        "prepared_mesh_topology_invalid", "Prepared mesh must be one connected triangulated disk.");
    const auto lower_coordinate = [&](std::uint32_t index) {
        const auto& edge = mesh.boundary_edges[index];
        const auto& a = mesh.nodes[edge.nodes[0]]; const auto& b = mesh.nodes[edge.nodes[1]];
        return edge.tag == 1 || edge.tag == 3 ? std::min(a.x, b.x) : std::min(a.y, b.y);
    };
    std::sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
        cancellation.Tick();
        return mesh.boundary_edges[a].tag < mesh.boundary_edges[b].tag ||
            (mesh.boundary_edges[a].tag == mesh.boundary_edges[b].tag && lower_coordinate(a) < lower_coordinate(b));
    });
    std::size_t cursor = 0U;
    for (int side = 1; side <= 4; ++side) {
        const bool horizontal = side == 1 || side == 3;
        double position = horizontal ? rectangle.xllcorner : rectangle.yllcorner;
        const double finish = horizontal ? right : top;
        while (cursor < order.size() && mesh.boundary_edges[order[cursor]].tag == side) {
            cancellation.Tick();
            const auto& edge = mesh.boundary_edges[order[cursor]];
            const auto& a = mesh.nodes[edge.nodes[0]]; const auto& b = mesh.nodes[edge.nodes[1]];
            Require(lower_coordinate(order[cursor]) == position, "prepared_mesh_boundary_invalid",
                "Prepared mesh boundary has a gap or overlapping intervals.");
            position = horizontal ? std::max(a.x, b.x) : std::max(a.y, b.y);
            ++cursor;
        }
        Require(position == finish, "prepared_mesh_boundary_invalid", "Prepared mesh boundary does not cover every rectangle side.");
    }
    // A positive oriented triangle chain whose internal edges cancel and whose
    // boundary covers this rectangle once has winding number one inside and
    // zero outside. Thus triangle interiors cannot overlap and no hole remains;
    // a quadratic intersection search is unnecessary. Exact shared endpoints,
    // strict orientation and the disk checks above are essential to this claim.
    cancellation.Check();
    return mesh;
}

} // namespace prepared_mesh_detail

// The caller must fingerprint these same bytes before admitting a manifest.
// This is a canonical prepared-asset reader, not a permissive general Gmsh importer.
// Cancellation callbacks run synchronously on the calling thread. An exception
// from a callback is reported as a read failure and never escapes this function.
[[nodiscard]] inline PreparedTriangleMeshReadResult ReadPreparedTriangleMesh(
    std::string_view bytes, PreparedMeshRectangle rectangle, PreparedTriangleMeshLimits limits = {},
    std::stop_token cancellation = {}, std::function<bool()> cancelled = {}) noexcept {
    try {
        prepared_mesh_detail::Cancellation control(cancellation, cancelled);
        auto mesh = prepared_mesh_detail::Read(bytes, rectangle, limits, control);
        return {std::move(mesh), "prepared_mesh_read", "Prepared triangular mesh read and validated."};
    } catch (const prepared_mesh_detail::Failure& error) {
        try { return {std::nullopt, error.code, error.message}; } catch (...) { return {}; }
    } catch (...) {
        try { return {std::nullopt, "prepared_mesh_read_failed", "Prepared mesh reading failed."}; } catch (...) { return {}; }
    }
}

} // namespace nsktsh::io
