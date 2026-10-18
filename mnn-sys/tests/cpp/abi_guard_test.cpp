#include "../../wrapper/mnn_abi_guard.h"

#include <cassert>
#include <stdexcept>
#include <string>

struct StringArrayResult {
    char** names;
    int count;
};

int main() {
    int* pointer_fallback = reinterpret_cast<int*>(0x1);
    int* pointer_result = mnn_rs_abi::with_c_boundary<int*>(nullptr, [=] {
        return pointer_fallback;
    });
    assert(pointer_result == pointer_fallback);
    assert(mnn_rs_abi::error_message() == nullptr);

    StringArrayResult array_result = mnn_rs_abi::with_c_boundary<StringArrayResult>(
        {nullptr, 0}, []() -> StringArrayResult {
            throw std::runtime_error("array construction failed");
        });
    assert(array_result.names == nullptr);
    assert(array_result.count == 0);
    assert(std::string(mnn_rs_abi::error_message()) == "array construction failed");

    float float_result = mnn_rs_abi::with_c_boundary<float>(0.0f, []() -> float {
        throw std::runtime_error("numeric operation failed");
    });
    assert(float_result == 0.0f);
    assert(std::string(mnn_rs_abi::error_message()) == "numeric operation failed");

    bool bool_result = mnn_rs_abi::with_c_boundary<bool>(false, []() -> bool {
        throw std::runtime_error("MNN allocation failed");
    });
    assert(!bool_result);
    assert(std::string(mnn_rs_abi::error_message()) == "MNN allocation failed");

    int status_result = mnn_rs_abi::with_c_boundary<int>(-1, []() -> int {
        throw 7;
    });
    assert(status_result == -1);
    assert(std::string(mnn_rs_abi::error_message()) == "unknown C++ exception");

    bool called_after_void_exception = false;
    mnn_rs_abi::with_c_boundary_void([&] {
        throw std::runtime_error("void operation failed");
    });
    assert(std::string(mnn_rs_abi::error_message()) == "void operation failed");

    int recovered = mnn_rs_abi::with_c_boundary<int>(-1, [&] {
        called_after_void_exception = true;
        return 0;
    });
    assert(recovered == 0);
    assert(called_after_void_exception);
    assert(mnn_rs_abi::error_message() == nullptr);
}
