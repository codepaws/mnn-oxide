/**
 * @file mnn_module_wrapper.cpp
 * @brief C wrapper for MNN Module/Expr API implementation
 */

#include "mnn_module_wrapper.h"
#include "mnn_abi_guard.h"

#ifdef MNN_MODULE_ENABLED
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#else
// Stub implementation when MNN_MODULE_ENABLED is not defined
#endif

#include <vector>
#include <cstring>
#include <algorithm>
#include <string>
#include <memory>
#include <limits>

#ifdef MNN_MODULE_ENABLED
using namespace MNN::Express;
#endif

/* ============================================================================
 * Helper functions
 * ============================================================================ */

static std::vector<std::string> to_string_vector(const char** arr, int count) {
    std::vector<std::string> result;
    if (!arr || count <= 0) return result;

    result.reserve(count);
    for (int i = 0; i < count; i++) {
        if (arr[i]) {
            result.push_back(std::string(arr[i]));
        }
    }
    return result;
}

static std::vector<int> to_int_vector(const int* arr, int count) {
    std::vector<int> result;
    if (!arr || count <= 0) return result;

    result.assign(arr, arr + count);
    return result;
}

/* ============================================================================
 * Module Functions
 * ============================================================================ */

MNNModule* mnn_module_load_from_file(
    const char** inputs, int input_count,
    const char** outputs, int output_count,
    const char* file_path,
    const MNNModuleConfig* config
) {
    return mnn_oxide_abi::with_c_boundary<MNNModule*>(nullptr, [&]() -> MNNModule* {
#ifdef MNN_MODULE_ENABLED
    if (!file_path || !inputs || !outputs || input_count <= 0 || output_count <= 0) {
        return nullptr;
    }

    auto input_names = to_string_vector(inputs, input_count);
    auto output_names = to_string_vector(outputs, output_count);

    Module::Config module_config;
    if (config) {
        module_config.dynamic = config->dynamic;
        module_config.shapeMutable = config->shape_mutable;
        module_config.rearrange = config->rearrange;
    }

    Module* module = Module::load(input_names, output_names, file_path, &module_config);
    return reinterpret_cast<MNNModule*>(module);
#else
    return nullptr;
#endif

    });
}

MNNModule* mnn_module_load_from_buffer(
    const char** inputs, int input_count,
    const char** outputs, int output_count,
    const uint8_t* buffer, size_t buffer_size,
    const MNNModuleConfig* config
) {
    return mnn_oxide_abi::with_c_boundary<MNNModule*>(nullptr, [&]() -> MNNModule* {
#ifdef MNN_MODULE_ENABLED
    if (!buffer || buffer_size == 0 || !inputs || !outputs || input_count <= 0 || output_count <= 0) {
        return nullptr;
    }

    auto input_names = to_string_vector(inputs, input_count);
    auto output_names = to_string_vector(outputs, output_count);

    Module::Config module_config;
    if (config) {
        module_config.dynamic = config->dynamic;
        module_config.shapeMutable = config->shape_mutable;
        module_config.rearrange = config->rearrange;
    }

    Module* module = Module::load(input_names, output_names, buffer, buffer_size, &module_config);
    return reinterpret_cast<MNNModule*>(module);
#else
    return nullptr;
#endif

    });
}

void mnn_module_destroy(MNNModule* module) {
    mnn_oxide_abi::with_c_boundary_void([&] {
#ifdef MNN_MODULE_ENABLED
    if (module) {
        delete reinterpret_cast<Module*>(module);
    }
#else
    (void)module;
#endif

    });
}

MNNModule* mnn_module_clone(const MNNModule* module, bool share_params) {
    return mnn_oxide_abi::with_c_boundary<MNNModule*>(nullptr, [&]() -> MNNModule* {
#ifdef MNN_MODULE_ENABLED
    if (!module) return nullptr;

    Module* cloned = Module::clone(reinterpret_cast<const Module*>(module), share_params);
    return reinterpret_cast<MNNModule*>(cloned);
#else
    return nullptr;
#endif

    });
}

MNNVar* mnn_module_forward(MNNModule* module, const MNNVar* input) {
    return mnn_oxide_abi::with_c_boundary<MNNVar*>(nullptr, [&]() -> MNNVar* {
#ifdef MNN_MODULE_ENABLED
    if (!module || !input) return nullptr;

    auto* m = reinterpret_cast<Module*>(module);
    auto* v = reinterpret_cast<const VARP*>(input);

    VARP result = m->forward(*v);
    return reinterpret_cast<MNNVar*>(new VARP(result));
#else
    return nullptr;
#endif

    });
}

MNNVar** mnn_module_forward_multi(MNNModule* module,
                                   const MNNVar** inputs, int input_count,
                                   int* output_count) {
    return mnn_oxide_abi::with_c_boundary<MNNVar**>(nullptr, [&]() -> MNNVar** {
    if (output_count) *output_count = 0;
#ifdef MNN_MODULE_ENABLED
    if (!module || !inputs || input_count <= 0 || !output_count) return nullptr;

    auto* m = reinterpret_cast<Module*>(module);

    std::vector<VARP> input_vars;
    input_vars.reserve(input_count);
    for (int i = 0; i < input_count; i++) {
        if (inputs[i]) {
            input_vars.push_back(*reinterpret_cast<const VARP*>(inputs[i]));
        }
    }

    auto output_vars = m->onForward(input_vars);
    if (output_vars.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return nullptr;

    std::unique_ptr<MNNVar*[]> result(new MNNVar*[output_vars.size()]());
    std::vector<std::unique_ptr<VARP>> owned_vars;
    owned_vars.reserve(output_vars.size());
    for (size_t i = 0; i < output_vars.size(); i++) {
        owned_vars.emplace_back(new VARP(output_vars[i]));
        result[i] = reinterpret_cast<MNNVar*>(owned_vars.back().get());
    }

    for (auto& owned_var : owned_vars) owned_var.release();
    *output_count = static_cast<int>(output_vars.size());
    return result.release();
#else
    if (output_count) *output_count = 0;
    return nullptr;
#endif

    });
}

void mnn_var_array_free(MNNVar** vars, int count) {
    mnn_oxide_abi::with_c_boundary_void([&] {
        if (vars == nullptr) return;
#ifdef MNN_MODULE_ENABLED
        for (int i = 0; i < count; ++i) {
            delete reinterpret_cast<VARP*>(vars[i]);
        }
#else
        (void)count;
#endif
        delete[] vars;
    });
}

int mnn_module_get_parameter_count(const MNNModule* module) {
    return mnn_oxide_abi::with_c_boundary<int>(0, [&]() -> int {
#ifdef MNN_MODULE_ENABLED
    if (!module) return 0;

    auto* m = reinterpret_cast<Module*>(const_cast<MNNModule*>(module));
    auto params = m->parameters();
    return static_cast<int>(params.size());
#else
    return 0;
#endif

    });
}

MNNVar** mnn_module_get_parameters(const MNNModule* module, int* count) {
    return mnn_oxide_abi::with_c_boundary<MNNVar**>(nullptr, [&]() -> MNNVar** {
    if (count) *count = 0;
#ifdef MNN_MODULE_ENABLED
    if (!module || !count) return nullptr;

    auto* m = reinterpret_cast<Module*>(const_cast<MNNModule*>(module));
    auto params = m->parameters();

    if (params.empty()) return nullptr;
    if (params.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return nullptr;

    std::unique_ptr<MNNVar*[]> result(new MNNVar*[params.size()]());
    std::vector<std::unique_ptr<VARP>> owned_vars;
    owned_vars.reserve(params.size());
    for (size_t i = 0; i < params.size(); i++) {
        owned_vars.emplace_back(new VARP(params[i]));
        result[i] = reinterpret_cast<MNNVar*>(owned_vars.back().get());
    }

    for (auto& owned_var : owned_vars) owned_var.release();
    *count = static_cast<int>(params.size());
    return result.release();
#else
    if (count) *count = 0;
    return nullptr;
#endif

    });
}

void mnn_module_set_training(MNNModule* module, bool is_training) {
    mnn_oxide_abi::with_c_boundary_void([&] {
#ifdef MNN_MODULE_ENABLED
    if (!module) return;

    auto* m = reinterpret_cast<Module*>(module);
    m->setIsTraining(is_training);
#else
    (void)module;
    (void)is_training;
#endif

    });
}

bool mnn_module_is_training(const MNNModule* module) {
    return mnn_oxide_abi::with_c_boundary<bool>(false, [&]() -> bool {
#ifdef MNN_MODULE_ENABLED
    if (!module) return false;

    auto* m = reinterpret_cast<Module*>(const_cast<MNNModule*>(module));
    return m->getIsTraining();
#else
    return false;
#endif

    });
}

const char* mnn_module_get_name(const MNNModule* module) {
    return mnn_oxide_abi::with_c_boundary<const char*>(nullptr, [&]() -> const char* {
#ifdef MNN_MODULE_ENABLED
    if (!module) return "";

    auto* m = reinterpret_cast<const Module*>(module);
    static thread_local std::string name_storage;
    name_storage = m->name();
    return name_storage.c_str();
#else
    return "";
#endif

    });
}

void mnn_module_set_name(MNNModule* module, const char* name) {
    mnn_oxide_abi::with_c_boundary_void([&] {
#ifdef MNN_MODULE_ENABLED
    if (!module || !name) return;

    auto* m = reinterpret_cast<Module*>(module);
    m->setName(std::string(name));
#else
    (void)module;
    (void)name;
#endif

    });
}

/* ============================================================================
 * Variable (Expr) Functions
 * ============================================================================ */

MNNVar* mnn_var_create_input(const int* shape, int dim_count,
                              MNNDimensionFormat format,
                              int type_code, int type_bits) {
    return mnn_oxide_abi::with_c_boundary<MNNVar*>(nullptr, [&]() -> MNNVar* {
#ifdef MNN_MODULE_ENABLED
    if (!shape || dim_count <= 0) return nullptr;

    auto shape_vec = to_int_vector(shape, dim_count);

    // Create input variable
    halide_type_t type{};
    type.code = static_cast<decltype(type.code)>(type_code);
    type.bits = static_cast<decltype(type.bits)>(type_bits);
    type.lanes = 1;
    VARP var = _Input(shape_vec, static_cast<Dimensionformat>(format), type);
    return reinterpret_cast<MNNVar*>(new VARP(var));
#else
    return nullptr;
#endif

    });
}

MNNVar* mnn_var_create_constant_float(const float* data,
                                       const int* shape, int dim_count,
                                       MNNDimensionFormat format) {
    return mnn_oxide_abi::with_c_boundary<MNNVar*>(nullptr, [&]() -> MNNVar* {
#ifdef MNN_MODULE_ENABLED
    if (!data || !shape || dim_count <= 0) return nullptr;

    auto shape_vec = to_int_vector(shape, dim_count);

    // Create constant variable from data
    VARP var = _Const(data, shape_vec, static_cast<Dimensionformat>(format));
    return reinterpret_cast<MNNVar*>(new VARP(var));
#else
    return nullptr;
#endif

    });
}

void mnn_var_destroy(MNNVar* var) {
    mnn_oxide_abi::with_c_boundary_void([&] {
#ifdef MNN_MODULE_ENABLED
    if (var) {
        delete reinterpret_cast<VARP*>(var);
    }
#else
    (void)var;
#endif

    });
}

MNNVarInfo* mnn_var_get_info(const MNNVar* var) {
    return mnn_oxide_abi::with_c_boundary<MNNVarInfo*>(nullptr, [&]() -> MNNVarInfo* {
#ifdef MNN_MODULE_ENABLED
    if (!var) return nullptr;

    auto* v = reinterpret_cast<const VARP*>(var);
    if (!v->get()) return nullptr;

    auto* variable = v->get();
    const auto* variable_info = variable->getInfo();
    if (!variable_info) return nullptr;

    auto shape = variable_info->dim;
    if (shape.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return nullptr;
    }

    std::unique_ptr<MNNVarInfo> info(new MNNVarInfo{});
    std::unique_ptr<int[]> ownedShape(new int[shape.size()]);
    std::copy(shape.begin(), shape.end(), ownedShape.get());

    info->dim_count = static_cast<int>(shape.size());
    auto type = variable_info->type;
    info->type_code = static_cast<int>(type.code);
    info->type_bits = type.bits;

    info->format = static_cast<MNNDimensionFormat>(variable_info->order);
    info->shape = ownedShape.release();

    return info.release();
#else
    return nullptr;
#endif

    });
}

void mnn_var_info_free(MNNVarInfo* info) {
    mnn_oxide_abi::with_c_boundary_void([&] {
    if (!info) return;

    if (info->shape) {
        delete[] info->shape;
    }
    delete info;

    });
}

float* mnn_var_read_float(const MNNVar* var, int* count) {
    return mnn_oxide_abi::with_c_boundary<float*>(nullptr, [&]() -> float* {
    if (count) *count = 0;
#ifdef MNN_MODULE_ENABLED
    if (!var || !count) return nullptr;

    auto* v = reinterpret_cast<const VARP*>(var);

    // Get info
    auto* variable = v->get();
    if (!variable) return nullptr;
    auto info = variable->getInfo();
    if (!info) return nullptr;

    size_t element_count = 1;
    for (int dim : info->dim) {
        if (dim < 0 || (dim != 0 && element_count > static_cast<size_t>(std::numeric_limits<int>::max()) / static_cast<size_t>(dim))) return nullptr;
        element_count *= static_cast<size_t>(dim);
    }
    if (element_count == 0 || element_count > static_cast<size_t>(std::numeric_limits<int>::max())) return nullptr;

    // Note: This is a simplified implementation
    // Actual reading requires proper tensor access
    std::unique_ptr<float[]> data(new float[element_count]);
    // In real implementation, we need to get the underlying tensor data
    // This requires MNN's internal API access
    memset(data.get(), 0, element_count * sizeof(float));

    *count = static_cast<int>(element_count);
    return data.release();
#else
    if (count) *count = 0;
    return nullptr;
#endif

    });
}

bool mnn_var_write_float(MNNVar* var, const float* data, int count) {
    return mnn_oxide_abi::with_c_boundary<bool>(false, [&]() -> bool {
#ifdef MNN_MODULE_ENABLED
    if (!var || !data || count <= 0) return false;

    auto* v = reinterpret_cast<VARP*>(var);
    // Note: Writing requires proper tensor access which is complex
    // This is a placeholder
    (void)v;
    return false;
#else
    return false;
#endif

    });
}

int* mnn_var_get_shape(const MNNVar* var, int* count) {
    return mnn_oxide_abi::with_c_boundary<int*>(nullptr, [&]() -> int* {
    if (count) *count = 0;
#ifdef MNN_MODULE_ENABLED
    if (!var || !count) return nullptr;

    auto* v = reinterpret_cast<const VARP*>(var);
    auto* variable = v->get();
    if (!variable) return nullptr;
    auto info = variable->getInfo();
    if (!info) return nullptr;
    if (info->dim.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return nullptr;
    auto shape = info->dim;

    std::unique_ptr<int[]> result(new int[shape.size()]);
    std::copy(shape.begin(), shape.end(), result.get());

    *count = static_cast<int>(shape.size());
    return result.release();
#else
    if (count) *count = 0;
    return nullptr;
#endif

    });
}

/* ============================================================================
 * Arithmetic Operations
 * ============================================================================ */

MNNVar* mnn_var_add(const MNNVar* a, const MNNVar* b) {
    return mnn_oxide_abi::with_c_boundary<MNNVar*>(nullptr, [&]() -> MNNVar* {
#ifdef MNN_MODULE_ENABLED
    if (!a || !b) return nullptr;

    auto* va = reinterpret_cast<const VARP*>(a);
    auto* vb = reinterpret_cast<const VARP*>(b);

    VARP result = *va + *vb;
    return reinterpret_cast<MNNVar*>(new VARP(result));
#else
    return nullptr;
#endif

    });
}

MNNVar* mnn_var_sub(const MNNVar* a, const MNNVar* b) {
    return mnn_oxide_abi::with_c_boundary<MNNVar*>(nullptr, [&]() -> MNNVar* {
#ifdef MNN_MODULE_ENABLED
    if (!a || !b) return nullptr;

    auto* va = reinterpret_cast<const VARP*>(a);
    auto* vb = reinterpret_cast<const VARP*>(b);

    VARP result = *va - *vb;
    return reinterpret_cast<MNNVar*>(new VARP(result));
#else
    return nullptr;
#endif

    });
}

MNNVar* mnn_var_mul(const MNNVar* a, const MNNVar* b) {
    return mnn_oxide_abi::with_c_boundary<MNNVar*>(nullptr, [&]() -> MNNVar* {
#ifdef MNN_MODULE_ENABLED
    if (!a || !b) return nullptr;

    auto* va = reinterpret_cast<const VARP*>(a);
    auto* vb = reinterpret_cast<const VARP*>(b);

    VARP result = *va * *vb;
    return reinterpret_cast<MNNVar*>(new VARP(result));
#else
    return nullptr;
#endif

    });
}

MNNVar* mnn_var_div(const MNNVar* a, const MNNVar* b) {
    return mnn_oxide_abi::with_c_boundary<MNNVar*>(nullptr, [&]() -> MNNVar* {
#ifdef MNN_MODULE_ENABLED
    if (!a || !b) return nullptr;

    auto* va = reinterpret_cast<const VARP*>(a);
    auto* vb = reinterpret_cast<const VARP*>(b);

    VARP result = *va / *vb;
    return reinterpret_cast<MNNVar*>(new VARP(result));
#else
    return nullptr;
#endif

    });
}

/* ============================================================================
 * Reduce Operations
 * ============================================================================ */

MNNVar* mnn_var_sum(const MNNVar* var, const int* axes, int axis_count, bool keep_dims) {
    return mnn_oxide_abi::with_c_boundary<MNNVar*>(nullptr, [&]() -> MNNVar* {
#ifdef MNN_MODULE_ENABLED
    if (!var) return nullptr;

    auto* v = reinterpret_cast<const VARP*>(var);
    std::vector<int> dims = to_int_vector(axes, axis_count);

    VARP result = _ReduceSum(*v, dims);
    if (keep_dims) {
        // Keep dims handling would require additional MNN API
    }

    return reinterpret_cast<MNNVar*>(new VARP(result));
#else
    return nullptr;
#endif

    });
}

MNNVar* mnn_var_mean(const MNNVar* var, const int* axes, int axis_count, bool keep_dims) {
    return mnn_oxide_abi::with_c_boundary<MNNVar*>(nullptr, [&]() -> MNNVar* {
#ifdef MNN_MODULE_ENABLED
    if (!var) return nullptr;

    auto* v = reinterpret_cast<const VARP*>(var);
    std::vector<int> dims = to_int_vector(axes, axis_count);

    VARP result = _ReduceMean(*v, dims);

    return reinterpret_cast<MNNVar*>(new VARP(result));
#else
    return nullptr;
#endif

    });
}

MNNVar* mnn_var_max(const MNNVar* var, const int* axes, int axis_count, bool keep_dims) {
    return mnn_oxide_abi::with_c_boundary<MNNVar*>(nullptr, [&]() -> MNNVar* {
#ifdef MNN_MODULE_ENABLED
    if (!var) return nullptr;

    auto* v = reinterpret_cast<const VARP*>(var);
    std::vector<int> dims = to_int_vector(axes, axis_count);

    VARP result = _ReduceMax(*v, dims);

    return reinterpret_cast<MNNVar*>(new VARP(result));
#else
    return nullptr;
#endif

    });
}

MNNVar* mnn_var_min(const MNNVar* var, const int* axes, int axis_count, bool keep_dims) {
    return mnn_oxide_abi::with_c_boundary<MNNVar*>(nullptr, [&]() -> MNNVar* {
#ifdef MNN_MODULE_ENABLED
    if (!var) return nullptr;

    auto* v = reinterpret_cast<const VARP*>(var);
    std::vector<int> dims = to_int_vector(axes, axis_count);

    VARP result = _ReduceMin(*v, dims);

    return reinterpret_cast<MNNVar*>(new VARP(result));
#else
    return nullptr;
#endif

    });
}

/* ============================================================================
 * Shape Operations
 * ============================================================================ */

MNNVar* mnn_var_reshape(const MNNVar* var, const int* shape, int dim_count) {
    return mnn_oxide_abi::with_c_boundary<MNNVar*>(nullptr, [&]() -> MNNVar* {
#ifdef MNN_MODULE_ENABLED
    if (!var || !shape || dim_count <= 0) return nullptr;

    auto* v = reinterpret_cast<const VARP*>(var);
    auto shape_vec = to_int_vector(shape, dim_count);

    VARP result = _Reshape(*v, shape_vec);
    return reinterpret_cast<MNNVar*>(new VARP(result));
#else
    return nullptr;
#endif

    });
}

MNNVar* mnn_var_transpose(const MNNVar* var, const int* perm, int dim_count) {
    return mnn_oxide_abi::with_c_boundary<MNNVar*>(nullptr, [&]() -> MNNVar* {
#ifdef MNN_MODULE_ENABLED
    if (!var || !perm || dim_count <= 0) return nullptr;

    auto* v = reinterpret_cast<const VARP*>(var);
    auto perm_vec = to_int_vector(perm, dim_count);

    VARP result = _Transpose(*v, perm_vec);
    return reinterpret_cast<MNNVar*>(new VARP(result));
#else
    return nullptr;
#endif

    });
}

MNNVar* mnn_var_concat(const MNNVar** vars, int var_count, int axis) {
    return mnn_oxide_abi::with_c_boundary<MNNVar*>(nullptr, [&]() -> MNNVar* {
#ifdef MNN_MODULE_ENABLED
    if (!vars || var_count <= 0) return nullptr;

    std::vector<VARP> var_list;
    var_list.reserve(var_count);
    for (int i = 0; i < var_count; i++) {
        if (vars[i]) {
            var_list.push_back(*reinterpret_cast<const VARP*>(vars[i]));
        }
    }

    VARP result = _Concat(var_list, axis);
    return reinterpret_cast<MNNVar*>(new VARP(result));
#else
    return nullptr;
#endif

    });
}

MNNVar** mnn_var_split(const MNNVar* var, int splits, int axis, int* output_count) {
    return mnn_oxide_abi::with_c_boundary<MNNVar**>(nullptr, [&]() -> MNNVar** {
    if (output_count) *output_count = 0;
#ifdef MNN_MODULE_ENABLED
    if (!var || !output_count || splits <= 0) return nullptr;

    auto* v = reinterpret_cast<const VARP*>(var);

    auto* variable = v->get();
    if (!variable) return nullptr;
    auto* info = variable->getInfo();
    if (!info || info->dim.empty()) return nullptr;

    int normalized_axis = axis;
    if (normalized_axis < 0) {
        normalized_axis += static_cast<int>(info->dim.size());
    }
    if (normalized_axis < 0 || normalized_axis >= static_cast<int>(info->dim.size())) {
        return nullptr;
    }
    const int axis_size = info->dim[static_cast<size_t>(normalized_axis)];
    if (axis_size <= 0 || axis_size % splits != 0) return nullptr;

    INTS split_sizes(static_cast<size_t>(splits), axis_size / splits);
    auto results = _Split(*v, std::move(split_sizes), axis);
    if (results.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return nullptr;

    std::unique_ptr<MNNVar*[]> result(new MNNVar*[results.size()]());
    std::vector<std::unique_ptr<VARP>> owned_vars;
    owned_vars.reserve(results.size());
    for (size_t i = 0; i < results.size(); i++) {
        owned_vars.emplace_back(new VARP(results[i]));
        result[i] = reinterpret_cast<MNNVar*>(owned_vars.back().get());
    }

    for (auto& owned_var : owned_vars) owned_var.release();
    *output_count = static_cast<int>(results.size());
    return result.release();
#else
    if (output_count) *output_count = 0;
    return nullptr;
#endif

    });
}
