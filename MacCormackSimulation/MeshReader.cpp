// MeshReader.cpp - Gmsh .msh v2.2 ASCII parser

#include "MeshReader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

std::string MeshReader::last_error_;
std::unordered_map<int, std::string> MeshReader::physical_names_;

// File-scope node map shared between ParseNodes and ParseElements
static std::unordered_map<int, int> s_gmsh_to_local;

const std::string& MeshReader::GetLastError() {
    return last_error_;
}

const std::unordered_map<int, std::string>& MeshReader::GetPhysicalNames() {
    return physical_names_;
}

bool MeshReader::ReadMsh(const std::string& filename, TriMesh& mesh) {
    last_error_.clear();
    physical_names_.clear();
    s_gmsh_to_local.clear();
    mesh.Clear();

    std::ifstream file(filename);
    if (!file.is_open()) {
        last_error_ = "Cannot open file: " + filename;
        return false;
    }

    // Boundary line segments: {node0, node1} -> physical_tag
    std::vector<std::pair<std::array<int, 2>, int>> boundary_lines;

    std::string line;
    while (std::getline(file, line)) {
        if (line.find("$MeshFormat") != std::string::npos) {
            double version;
            int file_type, data_size;
            file >> version >> file_type >> data_size;
            std::getline(file, line);

            if (version < 2.0 || version >= 3.0) {
                last_error_ = "Unsupported mesh format version: " + std::to_string(version);
                return false;
            }
            if (file_type != 0) {
                last_error_ = "Only ASCII format supported (file_type must be 0)";
                return false;
            }
            std::getline(file, line); // $EndMeshFormat
        }
        else if (line.find("$PhysicalNames") != std::string::npos) {
            if (!ParsePhysicalNames(file)) return false;
        }
        else if (line.find("$Nodes") != std::string::npos) {
            if (!ParseNodes(file, mesh)) return false;
        }
        else if (line.find("$Elements") != std::string::npos) {
            if (!ParseElements(file, mesh, boundary_lines)) return false;
        }
    }

    if (mesh.NumNodes() == 0) {
        last_error_ = "No nodes found in mesh file";
        return false;
    }
    if (mesh.NumElements() == 0) {
        last_error_ = "No triangle elements found in mesh file";
        return false;
    }

    mesh.BuildEdges();
    mesh.ComputeGeometry();

    int fixed = mesh.FixOrientation();
    if (fixed > 0) {
        std::cerr << "MeshReader: Fixed orientation of " << fixed << " elements" << std::endl;
        mesh.ComputeGeometry();
    }

    AssignBoundaryTags(mesh, boundary_lines);
    s_gmsh_to_local.clear(); // Release memory

    return true;
}

bool MeshReader::ParsePhysicalNames(std::ifstream& file) {
    int num_names;
    file >> num_names;

    std::string line;
    std::getline(file, line);

    for (int i = 0; i < num_names; ++i) {
        int dimension, tag;
        std::string name;
        file >> dimension >> tag;

        std::getline(file, line);
        auto pos1 = line.find('"');
        auto pos2 = line.rfind('"');
        if (pos1 != std::string::npos && pos2 != pos1) {
            name = line.substr(pos1 + 1, pos2 - pos1 - 1);
        } else {
            std::istringstream iss(line);
            iss >> name;
        }

        // Key: dimension * 100000 + tag (avoids collision for tags up to 99999)
        physical_names_[dimension * 100000 + tag] = name;
    }

    std::getline(file, line); // $EndPhysicalNames
    return true;
}

bool MeshReader::ParseNodes(std::ifstream& file, TriMesh& mesh) {
    int num_nodes;
    file >> num_nodes;

    if (num_nodes <= 0) {
        last_error_ = "Invalid number of nodes: " + std::to_string(num_nodes);
        return false;
    }

    mesh.Reserve(num_nodes, 0);
    s_gmsh_to_local.reserve(num_nodes);

    std::string line;
    std::getline(file, line);

    for (int i = 0; i < num_nodes; ++i) {
        int gmsh_id;
        double x, y, z;
        file >> gmsh_id >> x >> y >> z;

        int local_id = mesh.AddNode(x, y);
        s_gmsh_to_local[gmsh_id] = local_id;
    }

    std::getline(file, line);
    std::getline(file, line); // $EndNodes
    return true;
}

bool MeshReader::ParseElements(std::ifstream& file, TriMesh& mesh,
                                std::vector<std::pair<std::array<int, 2>, int>>& boundary_lines) {
    int num_elements;
    file >> num_elements;

    std::string line;
    std::getline(file, line);

    for (int i = 0; i < num_elements; ++i) {
        int gmsh_id, elem_type, num_tags;
        file >> gmsh_id >> elem_type >> num_tags;

        int physical_tag = 0;
        for (int t = 0; t < num_tags; ++t) {
            int tag;
            file >> tag;
            if (t == 0) physical_tag = tag;
        }

        if (elem_type == GMSH_LINE) {
            int n0, n1;
            file >> n0 >> n1;

            auto it0 = s_gmsh_to_local.find(n0);
            auto it1 = s_gmsh_to_local.find(n1);
            if (it0 != s_gmsh_to_local.end() && it1 != s_gmsh_to_local.end()) {
                boundary_lines.push_back({{it0->second, it1->second}, physical_tag});
            }
        }
        else if (elem_type == GMSH_TRIANGLE) {
            int n0, n1, n2;
            file >> n0 >> n1 >> n2;

            auto it0 = s_gmsh_to_local.find(n0);
            auto it1 = s_gmsh_to_local.find(n1);
            auto it2 = s_gmsh_to_local.find(n2);
            if (it0 != s_gmsh_to_local.end() && it1 != s_gmsh_to_local.end()
                && it2 != s_gmsh_to_local.end()) {
                mesh.AddElement(it0->second, it1->second, it2->second, physical_tag);
            }
        }
        else {
            // Skip remaining node IDs for unknown element types
            // Consume the rest of the line to avoid stream desynchronization
            std::string rest;
            std::getline(file, rest);
        }
    }

    std::getline(file, line);
    std::getline(file, line); // $EndElements
    return true;
}

void MeshReader::AssignBoundaryTags(TriMesh& mesh,
                                     const std::vector<std::pair<std::array<int, 2>, int>>& boundary_lines) {
    std::unordered_map<int64_t, int> line_tags;
    for (const auto& bl : boundary_lines) {
        int n0 = bl.first[0], n1 = bl.first[1];
        if (n0 > n1) std::swap(n0, n1);
        int64_t key = (static_cast<int64_t>(n0) << 32) | static_cast<int64_t>(n1);
        line_tags[key] = bl.second;
    }

    for (int i = 0; i < mesh.NumEdges(); ++i) {
        const auto& edge = mesh.GetEdge(i);
        if (!edge.IsBoundary()) continue;

        int n0 = edge.nodes[0], n1 = edge.nodes[1];
        if (n0 > n1) std::swap(n0, n1);
        int64_t key = (static_cast<int64_t>(n0) << 32) | static_cast<int64_t>(n1);

        auto it = line_tags.find(key);
        if (it != line_tags.end()) {
            mesh.SetBoundaryTag(i, it->second);
        }
    }
}
