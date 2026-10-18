/*
 * Internal helpers for keeping C++ exceptions inside the MNN C ABI shim.
 * This file is part of the Rust wrapper; it does not alter MNN licensing.
 */
#ifndef MNN_RS_ABI_GUARD_H
#define MNN_RS_ABI_GUARD_H

#include <exception>
#include <string>
#include <utility>

namespace mnn_rs_abi {

inline std::string& last_error() {
    static thread_local std::string value;
    return value;
}

inline bool& last_error_unavailable() {
    static thread_local bool value = false;
    return value;
}

inline void clear_error() noexcept {
    last_error().clear();
    last_error_unavailable() = false;
}

inline void record_error(const char* message) noexcept {
    try {
        last_error().assign(message != nullptr ? message : "unknown C++ exception");
    } catch (...) {
        last_error().clear();
        last_error_unavailable() = true;
    }
}

inline const char* error_message() noexcept {
    if (last_error_unavailable()) {
        return "native operation failed; error detail could not be retained";
    }
    return last_error().empty() ? nullptr : last_error().c_str();
}

template <typename Result, typename Function>
Result with_c_boundary(Result fallback, Function&& function) noexcept {
    clear_error();
    try {
        return std::forward<Function>(function)();
    } catch (const std::exception& error) {
        record_error(error.what());
    } catch (...) {
        record_error("unknown C++ exception");
    }
    return fallback;
}

template <typename Function>
void with_c_boundary_void(Function&& function) noexcept {
    clear_error();
    try {
        std::forward<Function>(function)();
    } catch (const std::exception& error) {
        record_error(error.what());
    } catch (...) {
        record_error("unknown C++ exception");
    }
}

}  // namespace mnn_rs_abi

#endif  // MNN_RS_ABI_GUARD_H
