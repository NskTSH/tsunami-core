// TriElement.h - Triangle element and edge structures for unstructured mesh
// Used by TriMesh and DG solver

#ifndef TRI_ELEMENT_H
#define TRI_ELEMENT_H

#include <array>
#include <cmath>

/// Triangle element with 3 nodes
struct TriElement {
    std::array<int, 3> nodes;     // Global node indices (counter-clockwise)
    std::array<int, 3> edges;     // Global edge indices
    std::array<int, 3> neighbors; // Neighbor element indices (-1 = boundary)

    double area;                  // Element area
    double cx, cy;                // Centroid coordinates

    // Outward normals for each edge (scaled by edge length)
    std::array<double, 3> nx;     // Normal x-component * edge_length
    std::array<double, 3> ny;     // Normal y-component * edge_length

    int physical_tag;             // Gmsh physical group tag

    TriElement()
        : nodes{-1, -1, -1}
        , edges{-1, -1, -1}
        , neighbors{-1, -1, -1}
        , area(0.0), cx(0.0), cy(0.0)
        , nx{0.0, 0.0, 0.0}
        , ny{0.0, 0.0, 0.0}
        , physical_tag(0) {}

    /// Compute area and centroid from node coordinates
    void ComputeGeometry(double x0, double y0,
                         double x1, double y1,
                         double x2, double y2) {
        // Signed area (positive if CCW)
        area = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));
        cx = (x0 + x1 + x2) / 3.0;
        cy = (y0 + y1 + y2) / 3.0;
    }

    /// Compute outward normals for each edge (assumes CCW orientation)
    /// Edge i connects node[i] -> node[(i+1)%3]
    void ComputeNormals(double x0, double y0,
                        double x1, double y1,
                        double x2, double y2) {
        const double coords[3][2] = {{x0, y0}, {x1, y1}, {x2, y2}};
        for (int i = 0; i < 3; ++i) {
            int j = (i + 1) % 3;
            double dx = coords[j][0] - coords[i][0];
            double dy = coords[j][1] - coords[i][1];
            // Outward normal (rotated 90 degrees clockwise for CCW triangle)
            nx[i] =  dy;
            ny[i] = -dx;
        }
    }

    bool IsDegenerate() const { return std::abs(area) < 1e-30; }
};

/// Edge connecting two nodes, shared by up to 2 elements
struct TriEdge {
    std::array<int, 2> nodes;     // Global node indices
    int elem_left;                // Left element index (normal points away from left)
    int elem_right;               // Right element index (-1 = boundary)

    double length;                // Edge length
    double mx, my;                // Edge midpoint
    double normal_x, normal_y;   // Unit outward normal (from left to right)

    int boundary_tag;             // 0 = interior, >0 = boundary physical tag

    TriEdge()
        : nodes{-1, -1}
        , elem_left(-1), elem_right(-1)
        , length(0.0), mx(0.0), my(0.0)
        , normal_x(0.0), normal_y(0.0)
        , boundary_tag(0) {}

    /// Compute geometry from node coordinates
    void ComputeGeometry(double x0, double y0, double x1, double y1) {
        double dx = x1 - x0;
        double dy = y1 - y0;
        length = std::sqrt(dx * dx + dy * dy);
        mx = 0.5 * (x0 + x1);
        my = 0.5 * (y0 + y1);
        if (length > 1e-30) {
            // Normal: rotate edge direction 90 degrees clockwise
            normal_x =  dy / length;
            normal_y = -dx / length;
        }
    }

    bool IsBoundary() const { return elem_right < 0; }
};

#endif // TRI_ELEMENT_H
