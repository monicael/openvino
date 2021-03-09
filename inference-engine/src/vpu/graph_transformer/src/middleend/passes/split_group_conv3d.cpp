// Copyright (C) 2019-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/middleend/pass_manager.hpp"
#include "ie_memcpy.h"
#include "vpu/model/data_contents/ie_blob_content.hpp"

namespace vpu {

namespace {

vpu::Data createSubWeights(
    const Model& model, const vpu::Data& weights, const int group, const std::string& postfix) {

    const auto weightsContent = weights->content();
    VPU_THROW_UNLESS(weightsContent != nullptr, "need weights content");

    const auto weightsDesc = weights->desc();
    const int KW = weightsDesc.dim(Dim::W);
    const int KH = weightsDesc.dim(Dim::H);
    const int KD = weightsDesc.dim(Dim::D);
    const int KI = weightsDesc.dim(Dim::C);
    const int KO = weightsDesc.dim(Dim::N);

    auto weightsPtr = weightsContent->get<fp16_t>();
    VPU_THROW_UNLESS(weightsPtr != nullptr, "cannot get weights data");

    const ie::SizeVector subWeightsDims = {
        static_cast<size_t>(KW), static_cast<size_t>(KH), static_cast<size_t>(KD),
        static_cast<size_t>(KI), static_cast<size_t>(KO)};
    const ie::TensorDesc subWeightsDesc(ie::Precision::FP16, subWeightsDims, ie::Layout::NCDHW);
    auto subWeightsBlob = ie::make_shared_blob<fp16_t>(subWeightsDesc);
    subWeightsBlob->allocate();

    auto subWeightsPtr = subWeightsBlob->buffer().as<fp16_t*>();

    const auto volumeElements = static_cast<size_t>(KW * KH * KD * KI * KO);
    const auto volumeBytes = volumeElements * sizeof(fp16_t);
    ie_memcpy(subWeightsPtr, volumeBytes, weightsPtr + group * volumeElements, volumeBytes);

    auto subWeights = model->duplicateData(weights, postfix,
                                           DataDesc(subWeightsDims),
                                           ieBlobContent(subWeightsBlob));

    return subWeights;
}

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model& model) {
    VPU_PROFILE(splitGroupConv3D);

    for (const auto& stage : model->getStages()) {
        if (stage->type() != StageType::ConvND) {
            continue;
        }

        using PV = typename ie::PropertyVector<unsigned int>;

        const auto pads_begin = stage->attrs().get<PV>("pads_begin");
        const auto pads_end   = stage->attrs().get<PV>("pads_end");
        const auto strides    = stage->attrs().get<PV>("strides");
        const auto dilations  = stage->attrs().get<PV>("dilations");
        const auto groups     = stage->attrs().get<int>("groups");
        const auto try_hw     = stage->attrs().get<int>("try_hw");

        const int kernelNDims = static_cast<int>(pads_begin.size());
        VPU_THROW_UNLESS(kernelNDims == pads_end.size(),
                         "wrong pads ndims=%lu, expected=%d", pads_end.size(), kernelNDims);
        VPU_THROW_UNLESS(kernelNDims == strides.size(),
                         "wrong strides ndims=%lu, expected=%d", strides.size(), kernelNDims);
        VPU_THROW_UNLESS(kernelNDims == dilations.size(),
                         "wrong dilations ndims=%lu, expected=%d", dilations.size(), kernelNDims);

        if (groups == 1 || kernelNDims != 3) {
            continue;
        }

        VPU_THROW_UNLESS(stage->numInputs() == 3, "wrong number of inputs: %d", stage->numInputs());
        VPU_THROW_UNLESS(stage->numOutputs() == 1, "wrong number of outputs: %d", stage->numOutputs());

        auto input = stage->input(0);
        auto weights = stage->input(1);
        auto biases = stage->input(2);
        auto output = stage->output(0);

        DataDesc inputDesc = input->desc();
        DataDesc outputDesc = output->desc();

        VPU_THROW_UNLESS(inputDesc.type() == outputDesc.type(),
                         "input and output types must equal, but: input type=%d, output type=%d",
                         inputDesc.type(), outputDesc.type());
        if (inputDesc.type() != DataType::FP16) {
            continue;
        }

        VPU_THROW_UNLESS(inputDesc.dimsOrder() == outputDesc.dimsOrder(),
                         "input and output dim orders must equal");
        if (inputDesc.dimsOrder() != DimsOrder::NCDHW) {
            continue;
        }

        const int I_N = inputDesc.dim(Dim::N);
        const int IC = inputDesc.dim(Dim::C);
        const int ID = inputDesc.dim(Dim::D);
        const int IH = inputDesc.dim(Dim::H);
        const int IW = inputDesc.dim(Dim::W);

        const int ON = outputDesc.dim(Dim::N);
        const int OC = outputDesc.dim(Dim::C);
        const int OD = outputDesc.dim(Dim::D);
        const int OH = outputDesc.dim(Dim::H);
        const int OW = outputDesc.dim(Dim::W);

        DataDesc weightsDesc = weights->desc();
        VPU_THROW_UNLESS(weightsDesc.type() == DataType::FP16, "wrong weights type: %d", weightsDesc.type());

        model->disconnectStage(stage);

        // Split the input tensor along the C axis in as many subtensors as there are groups
        DataVector subInputs3D(groups);
        DataVector subOutputs3D(groups);
        const int inChPerGroup = IC / groups;
        const int outChPerGroup = OC / groups;

        for (int g = 0; g < groups; ++g) {
            const auto postfix = formatString("@group-conv3d(g=%d/%d)", g + 1, groups);
            const DataDesc subInputsDesc3D(inputDesc.type(), DimsOrder::NCDHW, {IW, IH, ID, inChPerGroup, I_N});
            subInputs3D[g] = model->duplicateData(input, postfix, subInputsDesc3D);
        }

        auto spliStage = _stageBuilder->addSplitStage(
            model,
            stage->name() + "@split-group-3d",
            stage->origLayer(),
            Dim::C,
            input,
            subInputs3D);

        // Create the new Conv3D layers with group = 1
        for (int idxG = 0; idxG < groups; ++idxG) {
            const auto postfix = formatString("@group-conv3d(g=%d/%d)", idxG + 1, groups);

            const DataDesc subOutputsDesc3D(inputDesc.type(), DimsOrder::NCDHW, {OW, OH, OD, outChPerGroup, ON});
            subOutputs3D[idxG] = model->duplicateData(output, postfix, subOutputsDesc3D);

            // Split biases

            auto subBias = model->addFakeData();
            if (biases->usage() != DataUsage::Fake) {
                const auto biasContent = biases->content();
                VPU_THROW_UNLESS(biasContent != nullptr, "bias content empty: ", stage->name());

                auto biasPtr = biasContent->get<fp16_t>();
                const auto biasPerGroupSz = static_cast<size_t>(outChPerGroup * sizeof(fp16_t));

                const ie::TensorDesc subBiasDesc(ie::Precision::FP16, {static_cast<size_t>(outChPerGroup)}, ie::Layout::C);
                auto subBiasBlob = ie::make_shared_blob<fp16_t>(subBiasDesc);
                subBiasBlob->allocate();
                auto subBiasPtr = subBiasBlob->buffer().as<fp16_t*>();

                ie_memcpy(subBiasPtr, biasPerGroupSz, biasPtr + idxG * outChPerGroup, biasPerGroupSz);

                subBias = model->duplicateData(biases, postfix,
                                               DataDesc({static_cast<size_t>(outChPerGroup)}),
                                               ieBlobContent(subBiasBlob));
            }

            // Split weights
            auto subWeights = createSubWeights(model, weights, idxG, postfix);

            // Add new Conv3D
            Stage conv3d = _stageBuilder->addConvNDStage(
                model,
                stage->name() + postfix,
                stage->origLayer(),
                subInputs3D[idxG],
                subOutputs3D[idxG],
                subWeights,
                subBias);

            conv3d->attrs().set("pads_begin", pads_begin);
            conv3d->attrs().set("pads_end",   pads_end);

            conv3d->attrs().set("strides",    strides);
            conv3d->attrs().set("dilations",  dilations);

            conv3d->attrs().set("groups",     1);

            conv3d->attrs().set("try_hw",     try_hw);
        }

        // Concatenate the outputs of the new Conv3D layers
        _stageBuilder->addConcatStage(
            model,
            stage->name() + "@concat-group",
            stage->origLayer(),
            Dim::C,
            subOutputs3D,
            output);

        model->removeStage(stage);
    }
}

} // namespace

Pass::Ptr PassManager::splitGroupConv3D() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

} // namespace vpu
