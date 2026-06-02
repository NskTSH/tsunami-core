// TriMesh.h - Unstructured triangular mesh for DG solver
// Stores nodes, elements, edges with connectivity and geometry

#ifndef TRI_MESH_H
#define TRI_MESH_H

#include "TriElement.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

/// Node in 2D mesh
struct MeshNode {
    double x, y;
    MeshNode() : x(0.0), y(0.0) {}
    MeshNode(double x_, double y_) : x(x_), y(y_) {}
};

/// Unstructured triangular mesh
class TriMesh {
public:
    TriMesh() = default;

    // ========== DATA ACCESS ==========

    int NumNodes() const { return static_cast<int>(nodes_.size()); }
    int NumElements() const { return static_cast<int>(elements_.size()); }
    int NumEdges() const { return static_cast<int>(edges_.size()); }
    int NumBoundaryEdges() const { return num_boundary_edges_; }

    const MeshNode& GetNode(int i) const { return nodes_[i]; }
    const TriElement& GetElement(int i) const { return elements_[i]; }
    const TriEdge& GetEdge(int i) const { return edges_[i]; }

    MeshNode& GetNode(int i) { return nodes_[i]; }
    TriElement& GetElement(int i) { return elements_[i]; }
    TriEdge& GetEdge(int i) { return edges_[i]; }

    const std::vector<MeshNode>& Nodes() const { return nodes_; }
    const std::vector<TriElement>& Elements() const { return elements_; }
    const std::vector<TriEdge>& Edges() const { return edges_; }

    /// Get edges belonging to a given boundary tag
    const std::vector<int>& GetBoundaryEdges(int tag) const;

    // ========== CONSTRUCTION ==========

    /// Clear all data (nodes, elements, edges, boundary info)
    void Clear();

    /// Reserve memory for expected sizes
    void Reserve(int num_nodes, int num_elements, int num_edges = 0);

    /// Add a node, returns its index
    int AddNode(double x, double y);

    /// Add a triangle element (node indices must be CCW), returns element index
    int AddElement(int n0, int n1, int n2, int physical_tag = 0);

    /// Build edge connectivity and geometry from existing nodes and elements.
    /// Must be called after all nodes and elements are added.
    void BuildEdges();

    /// Compute geometry (area, centroids, normals) for all elements and edges
    void ComputeGeometry();

    /// Mark boundary edges by physical tag
    void SetBoundaryTag(int edge_idx, int tag);

    // ========== VALIDATION ==========

    struct ValidationResult {
        bool valid = true;
        int degenerate_elements = 0;
        int negative_area_elements = 0;  // Wrong orientation (CW instead of CCW)
        int disconnected_nodes = 0;
        std::string message;
    };

    /// Validate mesh integrity
    ValidationResult Validate() const;

    /// Fix orientation: flip elements with negative area to CCW
    int FixOrientation();

    // ========== I/O ==========

    /// Print mesh summary to stderr
    void PrintSummary() const;

private:
    std::vector<MeshNode> nodes_;
    std::vector<TriElement> elements_;
    std::vector<TriEdge> edges_;

    int num_boundary_edges_ = 0;

    // Boundary tag -> list of edge indices
    std::unordered_map<int, std::vector<int>> boundary_edges_;

    // Edge lookup: maps sorted (n0, n1) pair to edge index
    std::unordered_map<int64_t, int> edge_map_;

    /// Canonical edge key from two node indices
    static int64_t EdgeKey(int n0, int n1);

    /// Find or create edge between two nodes
    int FindOrCreateEdge(int n0, int n1, int elem_idx, int local_edge_idx);
};

#endif // TRI_MESH_H
