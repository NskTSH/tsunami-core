// ExchangeEngine.cpp - Реализация координатора обменов

#include "ExchangeEngine.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <cmath>

void ExchangeEngine::RegisterPatch(int patch_id, std::shared_ptr<IPatch> patch) {
    patches_[patch_id] = patch;
    // Invalidate cache so it gets rebuilt on next exchange
    interface_cache_.clear();
}

void ExchangeEngine::RegisterCoupling(std::shared_ptr<CouplingMap> coupling) {
    couplings_.push_back(coupling);
}

void ExchangeEngine::ExchangeFieldsPhased(double dt, double t, ExchangePhase phase) {
    auto fields = GetFieldsForPhase(phase);
    ExchangeFields(dt, t, fields);
}

void ExchangeEngine::ExchangeFields(double dt, double t, const std::vector<FieldType>& fields) {
    // Перестроить кеш если он пуст (первый вызов или после Clear)
    if (interface_cache_.empty() && !patches_.empty()) {
        RebuildInterfaceCache();
    }

    // Фаза 1: PrepareExchange у всех патчей
    for (auto& [id, patch] : patches_) {
        patch->PrepareExchange(dt, t);
    }

    // Фаза 2: Заполнить буферы интерфейсов из патчей
    for (auto& [id, patch] : patches_) {
        auto interfaces = patch->GetInterfaces();
        for (const auto& iface : interfaces) {
            FillBufferFromPatch(iface.interface_id, patch, fields);
        }
    }

    // Фаза 3: Применить все CouplingMap (интерполяция)
    for (auto& coupling : couplings_) {
        ApplyCoupling(coupling, fields);
    }

    // Фаза 4: Скопировать из буферов обратно в патчи
    for (auto& [id, patch] : patches_) {
        auto interfaces = patch->GetInterfaces();
        for (const auto& iface : interfaces) {
            CopyBufferToPatch(iface.interface_id, patch, fields);
        }
    }

    // Фаза 5: ApplyExchange у всех патчей
    for (auto& [id, patch] : patches_) {
        patch->ApplyExchange(dt, t);
    }
}

void ExchangeEngine::Clear() {
    patches_.clear();
    couplings_.clear();
    interface_buffers_.clear();
    interface_cache_.clear();
}

// ========== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ==========

void ExchangeEngine::RebuildInterfaceCache() {
    interface_cache_.clear();
    for (auto& [id, patch] : patches_) {
        auto interfaces = patch->GetInterfaces();
        for (auto& iface : interfaces) {
            interface_cache_[iface.interface_id] = std::move(iface);
        }
    }
}

std::vector<FieldType> ExchangeEngine::GetFieldsForPhase(ExchangePhase phase) const {
    if (phase == ExchangePhase::AfterHalfStep) {
        return { FieldType::H_hat, FieldType::U_hat, FieldType::V_hat, FieldType::Eta_hat };
    } else {
        return { FieldType::H, FieldType::U, FieldType::V, FieldType::Eta };
    }
}

void ExchangeEngine::FillBufferFromPatch(int interface_id,
                                         std::shared_ptr<IPatch> patch,
                                         const std::vector<FieldType>& fields) {
    const PatchInterface* iface = FindInterface(interface_id);
    if (!iface) {
        return;
    }

    size_t num_points = iface->GetNumPoints();
    if (num_points == 0) {
        return;
    }

    InterfaceBuffer& buffer = interface_buffers_[interface_id];

    for (FieldType field : fields) {
        Real* field_data = patch->GetFieldData(field);
        if (!field_data) {
            continue;
        }

        std::vector<double>& field_buffer = buffer.GetFieldBuffer(field, num_points);

        for (size_t i = 0; i < num_points; ++i) {
            int local_idx = iface->points[i].local_index;
            if (local_idx >= 0) {
                field_buffer[i] = field_data[local_idx];
            }
        }
    }
}

void ExchangeEngine::CopyBufferToPatch(int interface_id,
                                       std::shared_ptr<IPatch> patch,
                                       const std::vector<FieldType>& fields) {
    const PatchInterface* iface = FindInterface(interface_id);
    if (!iface) {
        return;
    }

    size_t num_points = iface->GetNumPoints();
    if (num_points == 0) {
        return;
    }

    auto it = interface_buffers_.find(interface_id);
    if (it == interface_buffers_.end()) {
        return;
    }

    InterfaceBuffer& buffer = it->second;

    for (FieldType field : fields) {
        // Use GetIncomingFieldData so that MeshPatch can route writes to its
        // dedicated incoming buffers (H_in_buf_ etc.) instead of the DG-export
        // buffers (H_buf_), preventing CopyBufferToPatch from corrupting the
        // solution that was just exported by FillBufferFromPatch.
        Real* field_data = patch->GetIncomingFieldData(field);
        if (!field_data) {
            continue;
        }

        auto field_it = buffer.fields.find(field);
        if (field_it == buffer.fields.end()) {
            continue;
        }

        const std::vector<double>& field_buffer = field_it->second;

        for (size_t i = 0; i < std::min(num_points, field_buffer.size()); ++i) {
            int write_idx = iface->points[i].GetWriteIndex();
            if (write_idx >= 0) {
                field_data[write_idx] = field_buffer[i];
            }
        }
    }
}

void ExchangeEngine::ApplyCoupling(std::shared_ptr<CouplingMap> coupling,
                                   const std::vector<FieldType>& fields) {
    int iface_a_id = coupling->GetInterfaceAId();
    int iface_b_id = coupling->GetInterfaceBId();

    auto it_a = interface_buffers_.find(iface_a_id);
    auto it_b = interface_buffers_.find(iface_b_id);

    if (it_a == interface_buffers_.end() || it_b == interface_buffers_.end()) {
        return;
    }

    InterfaceBuffer& buffer_a = it_a->second;
    InterfaceBuffer& buffer_b = it_b->second;

    for (FieldType field : fields) {
        auto field_it_a = buffer_a.fields.find(field);
        auto field_it_b = buffer_b.fields.find(field);

        if (field_it_a == buffer_a.fields.end() || field_it_b == buffer_b.fields.end()) {
            continue;
        }

        std::vector<double>& data_a = field_it_a->second;
        std::vector<double>& data_b = field_it_b->second;

        // FIX: Save copies BEFORE zeroing to prevent data loss in bidirectional exchange
        std::vector<double> data_a_copy(data_a);
        std::vector<double> data_b_copy(data_b);

        // Zero target buffers (Transfer uses +=)
        std::fill(data_b.begin(), data_b.end(), 0.0);
        std::fill(data_a.begin(), data_a.end(), 0.0);

        // Transfer from original copies
        coupling->TransferAtoB(data_a_copy.data(), data_b.data());
        coupling->TransferBtoA(data_b_copy.data(), data_a.data());
    }
}

const PatchInterface* ExchangeEngine::FindInterface(int interface_id) const {
    // FIX: Search in cache (stable storage) instead of temporary vector
    auto it = interface_cache_.find(interface_id);
    if (it != interface_cache_.end()) {
        return &(it->second);
    }
    return nullptr;
}

// ========== InterfaceBuffer ==========

std::vector<double>& ExchangeEngine::InterfaceBuffer::GetFieldBuffer(FieldType field, size_t size) {
    auto it = fields.find(field);
    if (it == fields.end()) {
        fields[field] = std::vector<double>(size, 0.0);
        return fields[field];
    }

    if (it->second.size() != size) {
        it->second.resize(size, 0.0);
    }

    return it->second;
}
