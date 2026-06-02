// TriMesh.cpp - Unstructured triangular mesh implementation

#include "TriMesh.h"
#include <algorithm>
#include <iostream>
#include <cassert>
#include <cmath>

// ========== DATA ACCESS ==========

const std::vector<int>& TriMesh::GetBoundaryEdges(int tag) const {
    static const std::vector<int> empty;
    auto it = boundary_edges_.find(tag);
    return (it != boundary_edges_.end()) ? it->second : empty;
}

// ========== CONSTRUCTION ==========

void TriMesh::Clear() {
    nodes_.clear();
    elements_.clear();
    edges_.clear();
    edge_map_.clear();
    boundary_edges_.clear();
    num_boundary_edges_ = 0;
}

void TriMesh::Reserve(int num_nodes, int num_elements, int num_edges) {
    nodes_.reserve(num_nodes);
    elements_.reserve(num_elements);
    if (num_edges > 0) {
        edges_.reserve(num_edges);
    }
}

int TriMesh::AddNode(double x, double y) {
    int idx = static_cast<int>(nodes_.size());
    nodes_.emplace_back(x, y);
    return idx;
}

int TriMesh::AddElement(int n0, int n1, int n2, int physical_tag) {
    int idx = static_cast<int>(elements_.size());
    TriElement elem;
    elem.nodes = {n0, n1, n2};
    elem.physical_tag = physical_tag;
    elements_.push_back(elem);
    return idx;
}

int64_t TriMesh::EdgeKey(int n0, int n1) {
    if (n0 > n1) std::swap(n0, n1);
    return (static_cast<int64_t>(n0) << 32) | static_cast<int64_t>(n1);
}

int TriMesh::FindOrCreateEdge(int n0, int n1, int elem_idx, int local_edge_idx) {
    int64_t key = EdgeKey(n0, n1);
    auto it = edge_map_.find(key);

    if (it != edge_map_.end()) {
        // Edge already exists - this element is the right neighbor
        int edge_idx = it->second;
        if (edges_[edge_idx].elem_right >= 0) {
            // Non-manifold edge: 3+ elements share this edge - skip
            std::cerr << "TriMesh: Non-manifold edge (" << n0 << "," << n1
                      << ") shared by 3+ elements" << std::endl;
        } else {
            edges_[edge_idx].elem_right = elem_idx;
        }
        return edge_idx;
    }

    // Create new edge
    int edge_idx = static_cast<int>(edges_.size());
    TriEdge edge;
    edge.nodes = {n0, n1};
    edge.elem_left = elem_idx;
    edge.elem_right = -1; // Will be set if shared
    edges_.push_back(edge);
    edge_map_[key] = edge_idx;
    return edge_idx;
}

void TriMesh::BuildEdges() {
    edges_.clear();
    edge_map_.clear();
    boundary_edges_.clear();
    num_boundary_edges_ = 0;

    // Reset element neighbors and edges before rebuild
    for (auto& elem : elements_) {
        elem.neighbors = {-1, -1, -1};
        elem.edges = {-1, -1, -1};
    }

    // Estimate: ~1.5 edges per element for interior mesh
    edges_.reserve(elements_.size() * 3 / 2);

    for (int ei = 0; ei < static_cast<int>(elements_.size()); ++ei) {
        auto& elem = elements_[ei];
        for (int k = 0; k < 3; ++k) {
            int n0 = elem.nodes[k];
            int n1 = elem.nodes[(k + 1) % 3];
            int edge_idx = FindOrCreateEdge(n0, n1, ei, k);
            elem.edges[k] = edge_idx;
        }
    }

    // Build neighbor connectivity from edges
    for (int edge_idx = 0; edge_idx < static_cast<int>(edges_.size()); ++edge_idx) {
        const auto& edge = edges_[edge_idx];
        int eL = edge.elem_left;
        int eR = edge.elem_right;

        if (eL >= 0 && eR >= 0) {
            // Interior edge: set neighbor for both elements
            auto& elemL = elements_[eL];
            auto& elemR = elements_[eR];

            for (int k = 0; k < 3; ++k) {
                if (elemL.edges[k] == edge_idx) elemL.neighbors[k] = eR;
                if (elemR.edges[k] == edge_idx) elemR.neighbors[k] = eL;
            }
        }
    }

    // Count boundary edges
    num_boundary_edges_ = 0;
    for (const auto& edge : edges_) {
        if (edge.IsBoundary()) ++num_boundary_edges_;
    }
}

void TriMesh::ComputeGeometry() {
    // Elements
    for (auto& elem : elements_) {
        const auto& p0 = nodes_[elem.nodes[0]];
        const auto& p1 = nodes_[elem.nodes[1]];
        const auto& p2 = nodes_[elem.nodes[2]];

        elem.ComputeGeometry(p0.x, p0.y, p1.x, p1.y, p2.x, p2.y);
        elem.ComputeNormals(p0.x, p0.y, p1.x, p1.y, p2.x, p2.y);
    }

    // Edges
    for (auto& edge : edges_) {
        const auto& p0 = nodes_[edge.nodes[0]];
        const auto& p1 = nodes_[edge.nodes[1]];
        edge.ComputeGeometry(p0.x, p0.y, p1.x, p1.y);

        // Orient normal: should point from elem_left to elem_right
        // Check that normal points away from left element centroid
        if (edge.elem_left >= 0) {
            const auto& elemL = elements_[edge.elem_left];
            double dot = (edge.mx - elemL.cx) * edge.normal_x
                       + (edge.my - elemL.cy) * edge.normal_y;
            if (dot < 0.0) {
                // Flip normal
                edge.normal_x = -edge.normal_x;
                edge.normal_y = -edge.normal_y;
            }
        }
    }
}

void TriMesh::SetBoundaryTag(int edge_idx, int tag) {
    edges_[edge_idx].boundary_tag = tag;
    boundary_edges_[tag].push_back(edge_idx);
}

// ========== VALIDATION ==========

TriMesh::ValidationResult TriMesh::Validate() const {
    ValidationResult result;

    // Check degenerate and negative-area elements
    for (int i = 0; i < static_cast<int>(elements_.size()); ++i) {
        const auto& elem = elements_[i];
        if (elem.IsDegenerate()) {
            ++result.degenerate_elements;
        }
        if (elem.area < 0.0) {
            ++result.negative_area_elements;
        }
    }

    // Check node connectivity
    std::vector<bool> used(nodes_.size(), false);
    for (const auto& elem : elements_) {
        for (int n : elem.nodes) {
            if (n >= 0 && n < static_cast<int>(nodes_.size())) {
                used[n] = true;
            }
        }
    }
    for (int i = 0; i < static_cast<int>(nodes_.size()); ++i) {
        if (!used[i]) ++result.disconnected_nodes;
    }

    // Build message
    if (result.degenerate_elements > 0) {
        result.valid = false;
        result.message += "Degenerate elements: " + std::to_string(result.degenerate_elements) + ". ";
    }
    if (result.negative_area_elements > 0) {
        result.valid = false;
        result.message += "CW-oriented elements: " + std::to_string(result.negative_area_elements) + ". ";
    }
    if (result.disconnected_nodes > 0) {
        result.message += "Disconnected nodes: " + std::to_string(result.disconnected_nodes) + ". ";
    }
    if (result.valid && result.disconnected_nodes == 0) {
        result.message = "Mesh OK.";
    }

    return result;
}

int TriMesh::FixOrientation() {
    int fixed = 0;
    for (auto& elem : elements_) {
        if (elem.area < 0.0) {
            // Swap nodes 1 and 2 to flip orientation
            std::swap(elem.nodes[1], elem.nodes[2]);
            std::swap(elem.edges[0], elem.edges[2]);
            std::swap(elem.neighbors[0], elem.neighbors[2]);
            elem.area = -elem.area;

            // Recompute normals
            const auto& p0 = nodes_[elem.nodes[0]];
            const auto& p1 = nodes_[elem.nodes[1]];
            const auto& p2 = nodes_[elem.nodes[2]];
            elem.ComputeNormals(p0.x, p0.y, p1.x, p1.y, p2.x, p2.y);
            ++fixed;
        }
    }
    return fixed;
}

// ========== I/O ==========

void TriMesh::PrintSummary() const {
    std::cerr << "=== TriMesh Summary ===" << std::endl;
    std::cerr << "  Nodes:          " << NumNodes() << std::endl;
    std::cerr << "  Elements:       " << NumElements() << std::endl;
    std::cerr << "  Edges:          " << NumEdges() << std::endl;
    std::cerr << "  Boundary edges: " << NumBoundaryEdges() << std::endl;

    if (!elements_.empty()) {
        double min_area = elements_[0].area, max_area = elements_[0].area;
        double total_area = 0.0;
        for (const auto& e : elements_) {
            min_area = std::min(min_area, e.area);
            max_area = std::max(max_area, e.area);
            total_area += e.area;
        }
        std::cerr << "  Total area:     " << total_area << std::endl;
        std::cerr << "  Min elem area:  " << min_area << std::endl;
        std::cerr << "  Max elem area:  " << max_area << std::endl;
        std::cerr << "  Area ratio:     " << max_area / std::max(min_area, 1e-30) << std::endl;
    }

    auto vr = Validate();
    std::cerr << "  Validation:     " << vr.message << std::endl;
    std::cerr << "=======================" << std::endl;
}
