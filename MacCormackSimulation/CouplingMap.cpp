// CouplingMap.cpp - Реализация карты связей между интерфейсами

#include "CouplingMap.h"

CouplingMap::CouplingMap(int interface_a_id, int interface_b_id)
    : interface_a_id_(interface_a_id),
      interface_b_id_(interface_b_id) {
}

void CouplingMap::AddCouplingAtoB(int src_idx, int tgt_idx, double weight) {
    a_to_b_.emplace_back(src_idx, tgt_idx, weight);
}

void CouplingMap::AddCouplingBtoA(int src_idx, int tgt_idx, double weight) {
    b_to_a_.emplace_back(src_idx, tgt_idx, weight);
}

void CouplingMap::AddSymmetricCoupling(int idx_a, int idx_b, double weight) {
    // A → B: индекс в A это src, индекс в B это tgt
    a_to_b_.emplace_back(idx_a, idx_b, weight);

    // B → A: индекс в B это src, индекс в A это tgt
    b_to_a_.emplace_back(idx_b, idx_a, weight);
}

void CouplingMap::TransferAtoB(const double* source_data, double* target_data) const {
    Transfer(source_data, target_data, a_to_b_);
}

void CouplingMap::TransferBtoA(const double* source_data, double* target_data) const {
    Transfer(source_data, target_data, b_to_a_);
}

void CouplingMap::Transfer(const double* source_data, double* target_data,
                           const std::vector<CouplingPoint>& couplings) const {
    // Применить все связи: target[tgt] += weight * source[src]
    for (const auto& cp : couplings) {
        if (source_data && target_data && cp.source_index >= 0 && cp.target_index >= 0) {
            target_data[cp.target_index] += cp.weight * source_data[cp.source_index];
        }
    }
}
