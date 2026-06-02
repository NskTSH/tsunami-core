// MeshReader.h - Gmsh .msh file parser (v2.2 ASCII)
// Reads triangular meshes with physical groups and boundary tags

#ifndef MESH_READER_H
#define MESH_READER_H

#include "TriMesh.h"
#include <string>
#include <unordered_map>

/// Parser for Gmsh .msh v2.2 ASCII format
class MeshReader {
public:
    /// Read a .msh file and populate a TriMesh
    /// @param filename Path to .msh file
    /// @param mesh Output mesh (will be cleared and rebuilt)
    /// @return true on success, false on error (message in GetLastError())
    static bool ReadMsh(const std::string& filename, TriMesh& mesh);

    /// Get last error message
    static const std::string& GetLastError();

    /// Physical group names from $PhysicalNames section
    /// Key: (dimension, tag), Value: name
    static const std::unordered_map<int, std::string>& GetPhysicalNames();

private:
    static std::string last_error_;
    static std::unordered_map<int, std::string> physical_names_;

    // Gmsh element type codes
    static constexpr int GMSH_LINE = 1;      // 2-node line
    static constexpr int GMSH_TRIANGLE = 2;  // 3-node triangle

    /// Parse $PhysicalNames section
    static bool ParsePhysicalNames(std::ifstream& file);

    /// Parse $Nodes section
    static bool ParseNodes(std::ifstream& file, TriMesh& mesh);

    /// Parse $Elements section (lines -> boundary tags, triangles -> elements)
    static bool ParseElements(std::ifstream& file, TriMesh& mesh,
                              std::vector<std::pair<std::array<int, 2>, int>>& boundary_lines);

    /// Assign boundary tags to mesh edges based on boundary line segments
    static void AssignBoundaryTags(TriMesh& mesh,
                                   const std::vector<std::pair<std::array<int, 2>, int>>& boundary_lines);
};

#endif // MESH_READER_H
