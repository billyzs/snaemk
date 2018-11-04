#pragma once
#include <cmath>
#include <type_traits>
namespace distance_fn {
    template<typename T1 = void, typename T2 = void>
    struct l1_norm {
        constexpr auto operator()(const T1 &p1, const T2 p2) const {
            using Ret = std::common_type_t<T1, T2>;
            return std::abs(Ret{p1} - Ret{p2});
        }
    };

    template<>
    struct l1_norm<void, void> {
        template<typename T1, typename T2>
        constexpr auto operator()(T1 &&p1, T2 &&p2) const {
            using Ret = std::common_type_t<T1, T2>;
            return std::abs(Ret{p1} - Ret{p2});
        }
    };
} // distance_fn
