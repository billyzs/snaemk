/// MIT License
///
/// Copyright (c) 2018 Billy Zhou
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to deal
/// in the Software without restriction, including without limitation the rights
/// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/// copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in all
/// copies or substantial portions of the Software.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
/// SOFTWARE.

#pragma once

// #include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <pstl/algorithm>
#include <pstl/execution>
#include <pstl/numeric>
#include <tuple>
#include <vector>

/// @brief compute max_iter counts of k-means on inputs ranging from i_begin to i_end, with initial
/// centroids ranging from o_begin to o_end
/// @tparam ExecutionPolicy the execution policy
/// @tparam InputIter iterator for input data
/// @tparam OutputIter iterator for centroids
/// @tparam DistanceFn function that computes how far an input data point is from a centroid; must have signature
/// RetType DistanceFn(const InputType&, const OutputType); RetType must have operator< defined; lesser value indicates
/// closer proximity to a centroid
/// @tparam AddOp how to accumulate InputIter to compute the new centroid
/// @param policy the execution policy to use
/// @param i_begin start of input range
/// @param i_end end of input range
/// @param[in,out] o_begin start of centroid range
/// @param[in,out] o_end end of centroid range
/// @param distance_metric measure of how close an input is to an centroid
/// @param AddOp how to accumulate centroid updates; defaults to std::plus for InputType
/// @param init_val initial value for centroid update; defaults to a default-constructed InputType
template <typename InputIter, typename OutputIter, typename DistanceFn,
          typename InputType = typename std::iterator_traits<InputIter>::value_type, typename ExecutionPolicy,
          typename AddOp = std::plus<InputType>>
std::pair<bool, std::vector<size_t>> k_means(
    ExecutionPolicy policy, const InputIter i_begin, const InputIter i_end, const OutputIter o_begin,
    const OutputIter o_end, const size_t max_iter,
    DistanceFn distance_metric, /** call with distance_metric(const InputType&, const OType&) */
    AddOp add = std::plus<InputType>(), InputType init_val = InputType{}) {
    using OutputType = typename std::iterator_traits<OutputIter>::value_type;  // must be constructable from InputType
    const auto num_input = static_cast<size_t>(std::distance(i_begin, i_end));
    const auto num_output = static_cast<size_t>(std::distance(o_begin, o_end));
    // associations[e] = c means the eth input is associated with the cth centroid
    std::vector<size_t> associations(num_input);
    std::vector<bool> convergence(num_input, false);  // bitset?
    std::iota(associations.begin(), associations.end(), 0);
    const auto indices = associations;
    std::vector<size_t> o_indices(num_output);
    std::iota(o_indices.begin(), o_indices.end(), 0);
    bool converged{false};

    // execute
    for (auto curr_iter = max_iter; !converged && curr_iter > 0; --curr_iter) {
        // assign association
        std::for_each(policy, indices.begin(), indices.end(), [&](const auto idx) {
            const InputType elem = *std::next(i_begin, idx);
            const auto centroid_choice = static_cast<size_t>(std::distance(
                o_begin, std::min_element(policy, o_begin, o_end, [&](const OutputType &c1, const OutputType &c2) {
                    // note: distance() not O(1) if iters aren't RandomAccess
                    return distance_metric(elem, c1) <= distance_metric(elem, c2);
                })));
            // if we were to use a single bool, it would have to be an atomic bool, leading to many atomic calls, which
            // might be slower than using a container of bools.
            convergence.at(idx) = (centroid_choice == std::exchange(associations.at(idx), centroid_choice));
        });
        converged = std::reduce(policy, convergence.cbegin(), convergence.cend(), true, std::bit_and<>());

        if (converged) break;

        // update centroid values
        std::for_each(policy, o_indices.cbegin(), o_indices.cend(), [&](const auto idx) {
            InputType new_centroid_val = init_val;
            // could do everything in one loop, but adding large numbers might overflow so count first & divide
            const double factor = 1.0f / std::count(policy, associations.cbegin(), associations.cend(), idx);
            // not parallelized because if it were, new_centroid_val has to be atomic
            std::for_each(indices.cbegin(), indices.cend(), [&](size_t i_idx) {
                if (associations.at(i_idx) == idx) {
                    new_centroid_val = add(
                        new_centroid_val, (*std::next(i_begin, i_idx)) * factor);  // assumes * is defined for InputType
                }
            });
            *std::next(o_begin, idx) = new_centroid_val;
        });
    }

    return std::make_pair(converged, std::move(associations));
}
