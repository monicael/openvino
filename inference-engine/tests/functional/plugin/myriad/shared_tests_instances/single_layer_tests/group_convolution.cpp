// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/group_convolution.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP16
};

/* ============= Group 3D Convolution ============= */

const std::vector<std::vector<size_t >> kernels3D = {{1, 3, 3}};
const std::vector<std::vector<size_t >> strides3D = {{1, 2, 2}};
const std::vector<std::vector<ptrdiff_t>> padBegins3D = {{0, 1, 1}};
const std::vector<std::vector<ptrdiff_t>> padEnds3D = {{0, 1, 1}};
const std::vector<std::vector<size_t >> dilations3D = {{1, 1, 1}};
const std::vector<size_t> numOutChannels3D = {24};
const std::vector<size_t> groups3D = {24, 1};

const auto conv3DParams_Group = ::testing::Combine(
        ::testing::ValuesIn(kernels3D),
        ::testing::ValuesIn(strides3D),
        ::testing::ValuesIn(padBegins3D),
        ::testing::ValuesIn(padEnds3D),
        ::testing::ValuesIn(dilations3D),
        ::testing::ValuesIn(numOutChannels3D),
        ::testing::ValuesIn(groups3D),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_CASE_P(smoke_Convolution3D_Group, GroupConvolutionLayerTest,
                        ::testing::Combine(
                                conv3DParams_Group,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t >({1, 24, 1, 24, 24})),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
                        GroupConvolutionLayerTest::getTestCaseName);

}  // namespace
