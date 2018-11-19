#include "distance-fns.hpp"
#include <snaemk/kmeans.hpp>
#include <snaemk/pkmeans.hpp>
#include <gtest/gtest.h>
#include <random>

using namespace distance_fn;

TEST(k_means, trivial_1) {
    constexpr std::array<float, 4> input{0,9,9,9};
    std::array<float, 2> centroid{1.0f, 8.0f};
    bool converged = false;
    std::tie(converged, std::ignore) =
            k_means(pstl::execution::par, input.cbegin(), input.cend(), centroid.begin(), centroid.end(), 3, l1_norm<>());
    ASSERT_TRUE(converged);
}

TEST(k_means, trivial_2) {
    constexpr std::array<float, 10> input{0,0,0,0,0,0,0,9,9,9};
    std::array<float, 2> centroid{1.0f, 8.0f};
    bool converged = false;
    std::vector<size_t> associations{0,0,0,0,0,0,0,0,0,0};
    const std::vector<size_t> expected_associations{0,0,0,0,0,0,0,1,1,1};
    std::tie(converged, associations) = k_means(pstl::execution::par, input.cbegin(), input.cend(),
                                                centroid.begin(), centroid.end(),
                                                std::numeric_limits<size_t>::max(),
                                                l1_norm<>());
    ASSERT_TRUE(converged);
    EXPECT_FLOAT_EQ(centroid[0], 0.0f);
    EXPECT_FLOAT_EQ(centroid[1], 9.0f);
    EXPECT_EQ(std::mismatch(associations.begin(), associations.end(),
                            expected_associations.begin(), expected_associations.end()),
              std::make_pair(associations.end(), expected_associations.end()));
}

TEST(k_means, moderate) {
    constexpr long long clusters = 4;
    constexpr long long bin_size = 10000;
    constexpr auto sample_size = clusters * bin_size;
    constexpr std::array<float, clusters> expected_centroid{10e9, 10e10, 10e11, 10e12};
    auto centroid = expected_centroid;
    std::array<float, sample_size> input;
    constexpr float percent_radius = 0.1f;
    std::random_device rd;
    // fill initial guesses of centroids, and input, with random data
    std::transform(centroid.begin(), centroid.end(), centroid.begin(),
            [&input, &rd, i_input = input.begin()](const float& f) mutable {
                std::uniform_real_distribution<float> rand(f*(1-percent_radius), f*(1+percent_radius));
                i_input = std::generate_n(i_input, bin_size, [&](){return rand(rd);});
                return rand(rd);
    });
    ASSERT_EQ(clusters*bin_size, input.size());

    // shuffle a few times for good measure
    std::shuffle(input.begin(), input.end(), std::mt19937(rd()));
    std::shuffle(input.begin(), input.end(), std::mt19937(rd()));
    std::shuffle(input.begin(), input.end(), std::mt19937(rd()));
    std::shuffle(input.begin(), input.end(), std::mt19937(rd()));
    std::shuffle(input.begin(), input.end(), std::mt19937(rd()));

    std::vector<size_t> association(clusters * bin_size);
    bool converged = false;
    std::tie(converged, association) = k_means(pstl::execution::par, input.cbegin(), input.cend(), centroid.begin(), centroid.end(), 20000,
            l1_norm<>());
    ASSERT_TRUE(converged);
    for (auto idx = 0; idx < clusters; idx++) {
        EXPECT_EQ(bin_size, static_cast<long long>(std::count(association.begin(), association.end(), idx)));
    }
    for (size_t idx = 0; idx < input.size(); idx++) {
        size_t actual = std::distance(expected_centroid.begin(),
                std::min_element(expected_centroid.cbegin(), expected_centroid.cend(),
                                 [elem = input.at(idx)](float c1, float c2){
                                    return l1_norm<>()(elem, c1) < l1_norm<>()(elem, c2);
        }));
        ASSERT_EQ(actual, association.at(idx)) << "association for " << idx <<"th input (" << input[idx] << ") incorrect";
    }
    for (size_t idx = 0; idx < centroid.size(); idx++) {
        EXPECT_LE(std::abs(centroid.at(idx) - expected_centroid.at(idx)) / expected_centroid.at(idx), percent_radius)
        << idx << "th centroid failed; expected " << expected_centroid.at(idx) << " but got " << centroid.at(idx) << "\n"
        << "centroids: " << centroid[0] << " " << centroid[1] << " " << centroid[2] << " " << centroid[3];
    }
}
