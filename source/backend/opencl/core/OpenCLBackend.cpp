//
//  OpenCLBackend.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/core/OpenCLBackend.hpp"
#include "MNN_generated.h"

#include "core/BufferAllocator.hpp"
#include "core/TensorUtils.hpp"
#include "shape/SizeComputer.hpp"
#include <map>
#include <mutex>
#include <thread>
#include "core/Macro.h"
#include "runtime/OpenCLTuneInfo.hpp"
//#define OPENCL_FALLBACK_LOG
namespace MNN {
namespace OpenCL {
#ifndef MNN_OPENCL_SEP_BUILD
void registerOpenCLOps();
#endif

CLRuntime::CLRuntime(const Backend::Info& info, int platformSize, int platformId, int deviceId){
    mInfo = info;

    BackendConfig::PrecisionMode precision = BackendConfig::Precision_Normal;
    BackendConfig::PowerMode power         = BackendConfig::Power_Normal;
    BackendConfig::MemoryMode memory       = BackendConfig::Memory_Normal;
    if (nullptr != mInfo.user) {
        precision = mInfo.user->precision;
        power     = mInfo.user->power;
        memory    = mInfo.user->memory;
    }

    // Shader precision
    mOpenCLRuntime.reset(new OpenCLRuntime(precision, mInfo.gpuMode, platformSize, platformId, deviceId));
    //Whether runtimeError
    mCLRuntimeError = mOpenCLRuntime->isCreateError();
    mPrecision = precision;
    mMemory = memory;
    mTunedInfo = new TuneInfo;
    
    mImagePool.reset(new ImagePool(mOpenCLRuntime->context()));
    mBufferPool.reset(new BufferPool(mOpenCLRuntime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR));
}

CLRuntime::~CLRuntime() {
    mImagePool = nullptr;
    mBufferPool = nullptr;
    mOpenCLRuntime = nullptr;
    delete mTunedInfo;
}
static bool _checkTensorInfo(const CLCache::TensorInfoT* dst, const Tensor* src) {
    if (dst->shape.size() != src->dimensions()) {
        return false;
    }
    for (int j=0; j<dst->shape.size(); ++j) {
        if (dst->shape[j] != src->length(j)) {
            return false;
        }
    }
    return true;
}

bool CLRuntime::onMeasure(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const MNN::Op* op, Runtime::OpInfo& dstInfo) const {
    dstInfo.initCostLong = true;
    if (nullptr == op->name()) {
        dstInfo.initCostLong = false;
        return true;
    }
    for(auto& info : mTunedInfo->mInfos) {
        if (info->type != op->type()) {
            continue;
        }
        if (info->name != op->name()->str()) {
            continue;
        }
        if (info->inputs.size() != inputs.size() || info->outputs.size() != outputs.size()) {
            continue;
        }
        bool match = true;
        for (int i=0; i<inputs.size(); ++i) {
            auto& dst = info->inputs[i];
            auto src = inputs[i];
            if (!_checkTensorInfo(dst.get(), src)) {
                match = false;
                break;
            }
        }
        if (!match) {
            continue;
        }
        for (int i=0; i<outputs.size(); ++i) {
            auto& dst = info->outputs[i];
            auto src = outputs[i];
            if (!_checkTensorInfo(dst.get(), src)) {
                match = false;
                break;
            }
        }
        if (match) {
            // All Info is match
            dstInfo.initCostLong = false;
            break;
        }
    }
    return true;
}


int CLRuntime::onGetRuntimeStatus(RuntimeStatus statusEnum) const {
    switch (statusEnum) {
        case STATUS_SUPPORT_FP16: {
            return mOpenCLRuntime->isDeviceSupportedFP16();
            break;
        }
        case STATUS_SUPPORT_DOT_PRODUCT: {
            return 0;
            break;
        }
        case STATUS_SUPPORT_POWER_LOW: {
            return mOpenCLRuntime->isDeviceSupportedLowPower();
            break;
        }
        default: {
            MNN_ERROR("unsupported interface");
            break;
        }
    }
    return 0;
}
void CLRuntime::onMaskOpReady(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           const MNN::Op* op) {
    if (nullptr != op->name()) {
        auto dstInfo = mTunedInfo;
        std::unique_ptr<CLCache::OpInfoT> opInfo(new CLCache::OpInfoT);;
        opInfo->type = op->type();
        opInfo->name = op->name()->str();
        opInfo->inputs.resize(inputs.size());
        for (int v=0; v<opInfo->inputs.size(); ++v) {
            opInfo->inputs[v].reset(new CLCache::TensorInfoT);
            opInfo->inputs[v]->shape.resize(inputs[v]->dimensions());
            for (int u=0; u<opInfo->inputs[v]->shape.size(); ++u) {
                opInfo->inputs[v]->shape[u] = inputs[v]->length(u);
            }
        }
        opInfo->outputs.resize(outputs.size());
        for (int v=0; v<opInfo->outputs.size(); ++v) {
            opInfo->outputs[v].reset(new CLCache::TensorInfoT);
            opInfo->outputs[v]->shape.resize(outputs[v]->dimensions());
            for (int u=0; u<opInfo->outputs[v]->shape.size(); ++u) {
                opInfo->outputs[v]->shape[u] = outputs[v]->length(u);
            }
        }
        dstInfo->mInfos.emplace_back(std::move(opInfo));
    }
}

bool CLRuntime::onSetCache(const void* buffer, size_t size) {
    if (nullptr == buffer) {
        return false;
    }
    auto cacheBuffer = CLCache::GetCache(buffer);
    flatbuffers::Verifier verify((const uint8_t*)buffer, size);
    if (false == CLCache::VerifyCacheBuffer(verify)) {
        return false;
    }
    if(nullptr != cacheBuffer->tuned()) {
        for (int i=0; i<cacheBuffer->tuned()->size(); ++i) {
            auto srcInfo = cacheBuffer->tuned()->GetAs<CLCache::OpInfo>(i);
            std::unique_ptr<CLCache::OpInfoT> dst(srcInfo->UnPack());
            mTunedInfo->mInfos.emplace_back(std::move(dst));
        }
    }
    bool res = mOpenCLRuntime->setCache(std::make_pair(buffer, size));
    
    #ifndef MNN_OPENCL_BUFFER_CLOSED
    if(mOpenCLRuntime->getGpuMemType() == BUFFER)
    {
        std::set<std::string> buildOptions;
        //when input or output need buffer2image transformation, open macro BUFFER_IMAGE_IO_TRANS
        //because cpu input and output are fp32
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
        if (mOpenCLRuntime->isSupportedIntelSubgroup()) {        
            mNCHWBufferToNC16HW16BufferInp = mOpenCLRuntime->buildKernel("buffer_convert_subgroup_buf", "nchw_buffer_to_nc16hw16_buffer_floatin", buildOptions);
            mNHWCBufferToNC16HW16BufferInp = mOpenCLRuntime->buildKernel("buffer_convert_subgroup_buf", "nhwc_buffer_to_nc16hw16_buffer_floatin", buildOptions);
            mNC4HW4BufferToNC16HW16BufferInp = mOpenCLRuntime->buildKernel("buffer_convert_subgroup_buf", "nc4hw4_buffer_to_nc16hw16_buffer_floatin", buildOptions);
            
            mNC16HW16BufferToNHWCBufferOut = mOpenCLRuntime->buildKernel("buffer_convert_subgroup_buf", "nc16hw16_buffer_to_nhwc_buffer_floatout", buildOptions);
            mNC16HW16BufferToNCHWBufferOut = mOpenCLRuntime->buildKernel("buffer_convert_subgroup_buf", "nc16hw16_buffer_to_nchw_buffer_floatout", buildOptions);
            mNC16HW16BufferToNC4HW4BufferOut = mOpenCLRuntime->buildKernel("buffer_convert_subgroup_buf", "nc16hw16_buffer_to_nc4hw4_buffer_floatout", buildOptions);
        }
#endif      
         mNCHWBufferToNC4HW4BufferInp = mOpenCLRuntime->buildKernel("buffer_convert_buf", "nchw_buffer_to_nc4hw4_buffer_floatin", buildOptions);
         mNHWCBufferToNC4HW4BufferInp = mOpenCLRuntime->buildKernel("buffer_convert_buf", "nhwc_buffer_to_nc4hw4_buffer_floatin", buildOptions);
         mNC4HW4BufferToNC4HW4BufferInp = mOpenCLRuntime->buildKernel("buffer_convert_buf", "nc4hw4_buffer_to_nc4hw4_buffer_floatin", buildOptions);

         mNC4HW4BufferToNHWCBufferOut = mOpenCLRuntime->buildKernel("buffer_convert_buf", "nc4hw4_buffer_to_nhwc_buffer_floatout", buildOptions);
         mNC4HW4BufferToNCHWBufferOut = mOpenCLRuntime->buildKernel("buffer_convert_buf", "nc4hw4_buffer_to_nchw_buffer_floatout", buildOptions);
         mNC4HW4BufferToNC4HW4BufferOut = mOpenCLRuntime->buildKernel("buffer_convert_buf", "nc4hw4_buffer_to_nc4hw4_buffer_floatout", buildOptions);

         mNC4HW4BufferToNC4HW4Buffer = mOpenCLRuntime->buildKernel("buffer_convert_buf", "nc4hw4_buffer_to_nc4hw4_buffer", buildOptions);
    }
    else
    #endif /* MNN_OPENCL_BUFFER_CLOSED */
    {
        std::set<std::string> buildOptions;
        //when input or output need buffer2image transformation, open macro BUFFER_IMAGE_IO_TRANS
        //because cpu input and output are fp32
        buildOptions.emplace("-DBUFFER_IMAGE_IO_TRANS");
        mNC4HW4BufferToImageFloat = mOpenCLRuntime->buildKernel("buffer_to_image", "nc4hw4_buffer_to_image", buildOptions);
        mNCHWBufferToImageFloat = mOpenCLRuntime->buildKernel("buffer_to_image", "nchw_buffer_to_image", buildOptions);
        mNHWCBufferToImageFloat = mOpenCLRuntime->buildKernel("buffer_to_image", "nhwc_buffer_to_image", buildOptions);
        mImageToNC4HW4BufferFloat = mOpenCLRuntime->buildKernel("buffer_to_image", "image_to_nc4hw4_buffer", buildOptions);
        mImageToNHWCBufferFloat = mOpenCLRuntime->buildKernel("buffer_to_image", "image_to_nhwc_buffer", buildOptions);
        mImageToNCHWBufferFloat = mOpenCLRuntime->buildKernel("buffer_to_image", "image_to_nchw_buffer", buildOptions);
    }
    return res;
}

std::pair<const void*, size_t> CLRuntime::onGetCache() {
    return mOpenCLRuntime->makeCache(mTunedInfo);
}

Backend* CLRuntime::onCreate(const BackendConfig* config) const {
    // FIXME: Use config info
    return new OpenCLBackend(mImagePool, mBufferPool, this);
}

void CLRuntime::onGabageCollect(int level) {
    mImagePool->releaseFreeList();
    mBufferPool->releaseFreeList();
}

float CLRuntime::onGetMemoryInMB() {
    auto staticMemoryInMB = mBufferPool->totalSize() / 1024.0f / 1024.0f;
    return staticMemoryInMB;
}

bool CLRuntime::isCLRuntimeError() {
    return mCLRuntimeError;
}

std::map<std::pair<OpType, GpuMemObject>, OpenCLBackend::Creator*>* gCreator() {
    static std::once_flag once;
    static std::map<std::pair<OpType, GpuMemObject>, OpenCLBackend::Creator*>* creators = nullptr;
    std::call_once(once, [&]() { creators = new std::map<std::pair<OpType, GpuMemObject>, OpenCLBackend::Creator*>; });
    return creators;
};

OpenCLBackend::OpenCLBackend(std::shared_ptr<ImagePool>imgPool, std::shared_ptr<BufferPool> bufPool, const CLRuntime *runtime)
    : Backend(MNN_FORWARD_OPENCL) {

    mCLRuntime = runtime;
    mOpenCLRuntime = mCLRuntime->mOpenCLRuntime;
    mPrecision = mCLRuntime->mPrecision;
    mMemory = mCLRuntime->mMemory;
    mStaticImagePool = imgPool;
    mStaticBufferPool = bufPool;
    if(mOpenCLRuntime.get()){
        if(mOpenCLRuntime->isCreateError() == true) {
            mIsCreateError = true;
        }

        mImagePool.reset(new ImagePool(mOpenCLRuntime->context()));
        mBufferPool.reset(new BufferPool(mOpenCLRuntime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR));
    }
    mMapMem = std::make_pair(0, nullptr);
    mUseRecordQueue = mOpenCLRuntime->isSupportRecordQueue();
    mDevideOpRecord = mOpenCLRuntime->isDevideOpRecord();
    mUseRecordableQueueSize = mOpenCLRuntime->getUseRecordableQueueSize();
}

OpenCLBackend::~OpenCLBackend() {
#ifdef LOG_VERBOSE
    MNN_PRINT("enter OpenCLBackend::~OpenCLBackend \n");
#endif
    releaseRecord();
    mRecordings.clear();
    mImagePool = nullptr;
    mBufferPool = nullptr;
    if(mMapMem.second != nullptr) {
    #ifdef MNN_OPENCL_SVM_ENABLE
        if(mUseSvm)
        {
            clSVMFree(mOpenCLRuntime->context().get(), mMapMem.second);
        }
        else
    #endif
        {
            free(mMapMem.second);
            mMapMem.second = nullptr;
        }
    }
}

OpenCLRuntime* OpenCLBackend::getOpenCLRuntime() {
    return mOpenCLRuntime.get();
}

class CLMemReleaseBuffer : public Backend::MemObj {
public:
    CLMemReleaseBuffer(cl::Buffer* bId, BufferPool* bufferPool) {
        mBuffer = bId;
        mBufferPool = bufferPool;
    }
    virtual ~ CLMemReleaseBuffer() {
        mBufferPool->recycle(mBuffer);
    }
private:
    cl::Buffer* mBuffer;
    BufferPool* mBufferPool;
};

class CLMemReleaseImage : public Backend::MemObj {
public:
    CLMemReleaseImage(cl::Image* bId, ImagePool* bufferPool) {
        mBuffer = bId;
        mBufferPool = bufferPool;
    }
    virtual ~ CLMemReleaseImage() {
        mBufferPool->recycle(mBuffer);
    }
private:
    cl::Image* mBuffer;
    ImagePool* mBufferPool;
};

Backend::MemObj* OpenCLBackend::onAcquire(const Tensor* nativeTensor, StorageType storageType) {
    #ifdef LOG_VERBOSE
    MNN_PRINT("Start OpenCLBackend::onAcquireBuffer !\n");
    #endif

    auto tensorShape = OpenCL::tensorShapeFormat(nativeTensor);
    int N = tensorShape.at(0);
    int H = tensorShape.at(1);
    int W = tensorShape.at(2);
    int C = tensorShape.at(3);

    #ifdef LOG_VERBOSE
    MNN_PRINT("OpenCLBackend::onAcquireBuffer: NHWC:[%d, %d, %d, %d]\n", N, H, W, C);
    #endif

    #ifndef MNN_OPENCL_BUFFER_CLOSED
    if(mOpenCLRuntime->getGpuMemType() == BUFFER) {
        size_t size;
        if (nativeTensor->dimensions() >= 2) {
            auto alignC = ROUND_UP(C, 8);
            // increment of height and width
            auto hR = ROUND_UP(H + 3, 4) - H;
            auto wR = ROUND_UP(W + 3, 4) - W;
            size = N * alignC * W * H;
            size = size + hR * W * 4 + wR * 4;
        } else {
            size = nativeTensor->elementSize();
            size = ROUND_UP(size, 4);
        }

        if (mOpenCLRuntime->isSupportedIntelSubgroup()) {
            int cPack = TensorUtils::getTensorChannelPack(nativeTensor);
            auto pads  = TensorUtils::getDescribe(nativeTensor)->mPads;
            size_t imageWidth  = (size_t) ROUND_UP(UP_DIV(C, cPack), 2) * ROUND_UP(pads.left + W + pads.right, 4);//C-round to 8,W-round to 4, for memory alloc
            size_t imageHeight = (size_t)N * H;
            size = imageWidth*imageHeight*cPack;
        }
        cl_channel_type dataType = CL_FLOAT;
        //when support and want fp16, use half datatype
        if (getOpenCLRuntime()->isSupportedFP16()) {
            dataType = CL_HALF_FLOAT;
        }

        if (storageType == DYNAMIC_SEPERATE) {
            auto buffer = mBufferPool->alloc(size*
                          (dataType==CL_HALF_FLOAT?sizeof(half_float::half):sizeof(float)), true);
            ((Tensor*)nativeTensor)->buffer().device = (uint64_t)buffer;
            return new CLMemReleaseBuffer(buffer, mBufferPool.get());
        }
        if (storageType == DYNAMIC) {
            auto buffer = mBufferPool->alloc(size*
                          (dataType==CL_HALF_FLOAT?sizeof(half_float::half):sizeof(float)));
            ((Tensor*)nativeTensor)->buffer().device = (uint64_t)buffer;
            return new CLMemReleaseBuffer(buffer, mBufferPool.get());
        }
        MNN_ASSERT(storageType == STATIC);
#ifdef MNN_LOW_MEMORY
        // for weight quant model's weight
        if ((nativeTensor->getType().code == halide_type_int) &&
            (nativeTensor->getType().bits == 8 || nativeTensor->getType().bits == 4)) {
            // int8 quant
            size_t alloc_size = size;
            if (nativeTensor->getType().bits == 4) {
                // int4 quant
                alloc_size = size / 2;
            }
            auto buffer = mStaticBufferPool->alloc(alloc_size);
            ((Tensor*)nativeTensor)->buffer().device = (uint64_t)buffer;
            return new CLMemReleaseBuffer(buffer, mStaticBufferPool.get());
        }
#endif
        auto buffer = mStaticBufferPool->alloc(size*
                     (dataType == CL_HALF_FLOAT ? sizeof(half_float::half) : sizeof(float)));
        ((Tensor*)nativeTensor)->buffer().device = (uint64_t)buffer; // fix
        return new CLMemReleaseBuffer(buffer, mStaticBufferPool.get());
    }
    else
    #endif /* MNN_OPENCL_BUFFER_CLOSED */
    {
        size_t imageWidth  = (size_t) (UP_DIV(C, 4) * W);//image mode only C pack to 4
        size_t imageHeight = (size_t)N * H;
        cl_channel_type dataType = CL_HALF_FLOAT;
        //when user want high precision, use float datatype
        if (mPrecision == BackendConfig::Precision_High) {
            dataType = CL_FLOAT;
        }

        if (storageType == DYNAMIC_SEPERATE) {
            auto image                               = mImagePool->alloc(imageWidth, imageHeight, dataType, true);
            ((Tensor*)nativeTensor)->buffer().device = (uint64_t)image; // fix
            return new CLMemReleaseImage(image, mImagePool.get());
        }
        if (storageType == DYNAMIC) {
            auto image                               = mImagePool->alloc(imageWidth, imageHeight, dataType);
            ((Tensor*)nativeTensor)->buffer().device = (uint64_t)image; // fix
            return new CLMemReleaseImage(image, mImagePool.get());
        }
        MNN_ASSERT(storageType == STATIC);
        auto image                               = mStaticImagePool->alloc(imageWidth, imageHeight, dataType);
        ((Tensor*)nativeTensor)->buffer().device = (uint64_t)image; // fix
        return new CLMemReleaseImage(image, mStaticImagePool.get());
    }
}

bool OpenCLBackend::onClearBuffer() {
    mImagePool->clear();
    mBufferPool->clear();
    if(mMapMem.second != nullptr) {
    #ifdef MNN_OPENCL_SVM_ENABLE
        if(mUseSvm)
        {
            clSVMFree(mOpenCLRuntime->context().get(), mMapMem.second);
        }
        else
    #endif
        {
            free(mMapMem.second);
            mMapMem.second = nullptr;
        }
    }
    return true;
}

Execution* OpenCLBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                   const MNN::Op* op) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start OpenCLBackend::onCreate \n");
#endif
    auto creators = gCreator();
    auto iter      = creators->find(std::make_pair(op->type(), mOpenCLRuntime->getGpuMemType()));
    if (0 != inputs.size() && (getDataType(inputs[0]) == DataType_DT_INT8 || inputs[0]->getType().bytes() == 1)) {
        #ifdef OPENCL_FALLBACK_LOG
        MNN_PRINT("Don't support type %s for int8 input\n", EnumNameOpType(op->type()));
        #endif
        return NULL;
    }
    if (iter == creators->end()) {
        mDevideOpRecord = true;
        #ifdef OPENCL_FALLBACK_LOG
        if (nullptr != op->name()) {
            MNN_PRINT("Don't support type %s memObject:%d, %s\n", EnumNameOpType(op->type()), mOpenCLRuntime->getGpuMemType(), op->name()->c_str());
        } else {
            MNN_PRINT("Don't support type %s memObject:%d\n", EnumNameOpType(op->type()), mOpenCLRuntime->getGpuMemType());
        }
        #endif
        return NULL;
    }

    if(mOpenCLRuntime->getGpuMemType() == IMAGE) {
        auto maxImageSize = mOpenCLRuntime->getMaxImage2DSize();
        bool valid        = true;
        for (auto t : inputs) {
            auto tensorShape = OpenCL::tensorShapeFormat(t);
            int imageHeight = tensorShape[0] * tensorShape[1];
            int imageWidth  = tensorShape[2] * UP_DIV(tensorShape[3], 4);
            if (imageHeight > maxImageSize.at(0) || imageWidth > maxImageSize.at(1)) {
                valid = false;
                break;
            }
        }
        for (auto t : outputs) {
            auto tensorShape = OpenCL::tensorShapeFormat(t);
            int imageHeight = tensorShape[0] * tensorShape[1];
            int imageWidth  = tensorShape[2] * UP_DIV(tensorShape[3], 4);
            if (imageHeight > maxImageSize.at(0) || imageWidth > maxImageSize.at(1)) {
                valid = false;
                break;
            }
        }

        if (!valid) {
            mDevideOpRecord = true;
            #ifdef OPENCL_FALLBACK_LOG
            for (auto t : inputs) {
                auto tensorShape = OpenCL::tensorShapeFormat(t);
                MNN_PRINT("input n:%d, h:%d, w:%d, c:%d\n", tensorShape[0], tensorShape[1], tensorShape[2], tensorShape[3]);
            }
            for (auto t : outputs) {
                auto tensorShape = OpenCL::tensorShapeFormat(t);
                MNN_PRINT("output n:%d, h:%d, w:%d, c:%d\n", tensorShape[0], tensorShape[1], tensorShape[2], tensorShape[3]);
            }
            MNN_PRINT("beyond cl_image creat size! fallback to cpu backend\n");
            #endif
            return NULL;
        }
    }

    auto exe = iter->second->onCreate(inputs, outputs, op, this);
    if (NULL == exe) {
        mDevideOpRecord = true;
        #ifdef OPENCL_FALLBACK_LOG
        if (nullptr != op->name()) {
            MNN_PRINT("The Creator Don't support type %s, memObject:%d, %s\n", MNN::EnumNameOpType(op->type()), mOpenCLRuntime->getGpuMemType(), op->name()->c_str());
        } else {
            MNN_PRINT("The Creator Don't support type %s, memObject:%d,\n", EnumNameOpType(op->type()), mOpenCLRuntime->getGpuMemType());
        }
        #endif
        return NULL;
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("End OpenCLBackend::onCreate \n");
#endif
    return exe;
}

void OpenCLBackend::onResizeBegin() {
#ifndef ENABLE_OPENCL_TIME_PROFILER
    mOpenCLRuntime->setCommandQueueProfileEnable();
#endif
    releaseRecord();
}

ErrorCode OpenCLBackend::onResizeEnd() {
#ifndef ENABLE_OPENCL_TIME_PROFILER
    mOpenCLRuntime->setCommandQueueProfileDisable();
#endif
    if(!mRecordings.empty()){
        endRecord(mRecordings.back(), true);
    }
    return NO_ERROR;
}

void OpenCLBackend::onExecuteBegin() const {
    mOpenCLRuntime->mQueueCount = 0;
    clearRecord();
    mOpenCLRuntime->clearEvent();
}

void OpenCLBackend::onExecuteEnd() const {
    mOpenCLRuntime->mQueueCount = 0;
    clearRecord();
    enqeueRecord();
    mOpenCLRuntime->printEventTime();
}


bool OpenCLBackend::isCreateError() const {
    return mIsCreateError;
}

void OpenCLBackend::_allocHostBuffer(int length) const {
    MNN_ASSERT(length > 0);
    if (nullptr != mHostBuffer.second && length <= mHostBuffer.first) {
        return;
    }
    mHostBuffer.first = length;
    mHostBuffer.second.reset(
        new cl::Buffer(mOpenCLRuntime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, length));
}

void OpenCLBackend::copyFromDeviceInt8(const Tensor* srcTensor, const Tensor* dstTensor) const{
    std::vector<int> bufferShape = MNN::OpenCL::tensorShapeFormat(dstTensor);


    auto needSize = dstTensor->size();
    auto hostPtr = dstTensor->host<int8_t>();
    auto DeviceBuffer = (cl::Buffer*)srcTensor->deviceId();
    cl_int error                = CL_SUCCESS;

#ifndef MNN_OCL_QUANT_DUMP
    error = mOpenCLRuntime->commandQueue().enqueueReadBuffer(*DeviceBuffer, CL_TRUE, 0, needSize, hostPtr);
    MNN_ASSERT(error == 0);
#else//for dump test
    int8_t* tmpPtr = (int8_t *)malloc(needSize);
    error = mOpenCLRuntime->commandQueue().enqueueReadBuffer(*DeviceBuffer, CL_TRUE, 0, needSize, tmpPtr);
    MNN_ASSERT(error == 0);
    int C_4 = (bufferShape[3]+3)/4;
    for(int n=0; n<bufferShape[0]; n++) {
        for(int c=0; c<bufferShape[3]; c++) {
            for(int h=0; h<bufferShape[1]; h++) {
                for(int w=0; w<bufferShape[2]; w++) {
                   hostPtr[n*bufferShape[3]*bufferShape[1]*bufferShape[2] + c*bufferShape[1]*bufferShape[2] + h*bufferShape[2] + w] =
                    tmpPtr[n*C_4*bufferShape[1]*bufferShape[2]*4 + (c/4)*bufferShape[1]*bufferShape[2]*4 + h*bufferShape[2]*4 + w*4 + c%4];
                }
            }
        }
    }
    if(tmpPtr != nullptr) {
        free(tmpPtr);
        tmpPtr = nullptr;
    }
#endif

#ifdef ENABLE_OPENCL_TIME_PROFILER
    MNN_PRINT("total kernel time:%d us\n", (int)mOpenCLRuntime->mKernelTime);
#endif
}

void OpenCLBackend::copyToDeviceInt8(const Tensor* srcTensor, const Tensor* dstTensor) const{
        auto needSize = srcTensor->size();
        auto hostPtr                = srcTensor->host<int8_t>();
        cl_int error                = CL_SUCCESS;
        auto DeviceBuffer = (cl::Buffer*)dstTensor->deviceId();
        mOpenCLRuntime->commandQueue().enqueueWriteBuffer(*DeviceBuffer, CL_TRUE, 0, needSize, hostPtr);
}
int OpenCLBackend::onSync(Tensor::MapType mtype, bool toCpu, const Tensor* dstTensor) {
    if (toCpu) {
        mOpenCLRuntime->commandQueue().finish();
    }
    return 0;
}

void CLRuntime::convertFromDevice(const Tensor* srcTensor, const Tensor* dstTensor, MNN_DATA_FORMAT data_format, bool svmFlag) const {
#ifndef MNN_OPENCL_BUFFER_CLOSED
    if(mOpenCLRuntime->getGpuMemType() == BUFFER)
    {
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
        int cPack = TensorUtils::getTensorChannelPack(srcTensor);
        if (cPack == 16 && mOpenCLRuntime->isSupportedIntelSubgroup()) {
            switch (data_format) {
                case MNN_DATA_FORMAT_NHWC:
                    OpenCL::convertNC4HW4OrNC16HW16BufferToNCHWOrNHWCBuffer(srcTensor, const_cast<Tensor*>(dstTensor),
                                                     *const_cast<cl::Kernel*>(&mNC16HW16BufferToNHWCBufferOut), "nc16hw16_buffer_to_nhwc_buffer", mOpenCLRuntime.get(), true, false, svmFlag);
                    break;
                case MNN_DATA_FORMAT_NCHW:
                    OpenCL::convertNC4HW4OrNC16HW16BufferToNCHWOrNHWCBuffer(srcTensor, const_cast<Tensor*>(dstTensor),
                                                     *const_cast<cl::Kernel*>(&mNC16HW16BufferToNCHWBufferOut), "nc16hw16_buffer_to_nchw_buffer", mOpenCLRuntime.get(), true, false, svmFlag);
                    break;
                case MNN_DATA_FORMAT_NC4HW4:
                    OpenCL::convertNC4HW4BufferBetweenNC16HW16Buffer(srcTensor, const_cast<Tensor*>(dstTensor),
                                                     *const_cast<cl::Kernel*>(&mNC16HW16BufferToNC4HW4BufferOut), "nc16hw16_buffer_to_nc4hw4_buffer", mOpenCLRuntime.get(), OutTrans, false, svmFlag, false, true);
                    break;
                default:
                    MNN_PRINT("output data format not support for subgroup!\n");
                    break;
            }
        } else 
#endif
        {
            switch (data_format) {
                case MNN_DATA_FORMAT_NHWC:
                    OpenCL::convertNC4HW4OrNC16HW16BufferToNCHWOrNHWCBuffer(srcTensor, const_cast<Tensor*>(dstTensor),
                                                     *const_cast<cl::Kernel*>(&mNC4HW4BufferToNHWCBufferOut), "nc4hw4_buffer_to_nhwc_buffer", mOpenCLRuntime.get(), true, false, svmFlag);
                    break;
                case MNN_DATA_FORMAT_NCHW:
                    OpenCL::convertNC4HW4OrNC16HW16BufferToNCHWOrNHWCBuffer(srcTensor, const_cast<Tensor*>(dstTensor),
                                                     *const_cast<cl::Kernel*>(&mNC4HW4BufferToNCHWBufferOut), "nc4hw4_buffer_to_nchw_buffer", mOpenCLRuntime.get(), true, false, svmFlag);
                    break;
                case MNN_DATA_FORMAT_NC4HW4:
                    OpenCL::convertNC4HW4BufferToNC4HW4Buffer(srcTensor, const_cast<Tensor*>(dstTensor),
                                                     *const_cast<cl::Kernel*>(&mNC4HW4BufferToNC4HW4BufferOut), mOpenCLRuntime.get(), OutTrans, false, svmFlag, false, true);
                    break;
                default:
                    MNN_PRINT("output data format not support!\n");
                    break;
            }
        }
    }
    else
#endif /* MNN_OPENCL_BUFFER_CLOSED */
    {
        switch (data_format) {
            case MNN_DATA_FORMAT_NHWC:
                OpenCL::convertImageToNHWCBuffer(srcTensor, const_cast<Tensor*>(dstTensor),
                                                 *const_cast<cl::Kernel*>(&mImageToNHWCBufferFloat), mOpenCLRuntime.get(), false, svmFlag);
                break;
            case MNN_DATA_FORMAT_NCHW:
                OpenCL::convertImageToNCHWBuffer(srcTensor, const_cast<Tensor*>(dstTensor),
                                                 *const_cast<cl::Kernel*>(&mImageToNCHWBufferFloat), mOpenCLRuntime.get(), false, svmFlag);
                break;
            case MNN_DATA_FORMAT_NC4HW4:
                OpenCL::convertImageToNC4HW4Buffer(srcTensor, const_cast<Tensor*>(dstTensor),
                                                   *const_cast<cl::Kernel*>(&mImageToNC4HW4BufferFloat), mOpenCLRuntime.get(), false, svmFlag);
                break;
            default:
                break;
        }
    }
}

void OpenCLBackend::copyFromDevice(const Tensor* srcTensor, const Tensor* dstTensor) const{
    auto needSize = dstTensor->size();

    void* hostPtr;
    void* tmpPtr;
    if(dstTensor->getType().code == halide_type_int) {
        if(dstTensor->getType().bits == 8){
            needSize *= 4;
            hostPtr = malloc(needSize);
        } else if(dstTensor->getType().bits == 32){
            hostPtr = malloc(needSize);
        } else {
            MNN_PRINT("opencl input datatype not support, bit:%d\n", dstTensor->getType().bits);
            MNN_ASSERT(false);
        }
    } else if(dstTensor->getType().code == halide_type_uint){
        if(dstTensor->getType().bits == 8){
            needSize *= 4;
            hostPtr = malloc(needSize);
        } else if(dstTensor->getType().bits == 32){
            hostPtr = malloc(needSize);
        } else {
            MNN_PRINT("opencl input datatype not support, bit:%d\n", dstTensor->getType().bits);
            MNN_ASSERT(false);
        }
    } else {
        hostPtr = dstTensor->host<float>();
    }

    _allocHostBuffer(needSize);

    MNN::Tensor interTensor(dstTensor, dstTensor->getDimensionType(), false);
    interTensor.buffer().device = (uint64_t)mHostBuffer.second.get();

    MNN_DATA_FORMAT data_format = TensorUtils::getDescribe(dstTensor)->dimensionFormat;
    
    //Convert format
    mCLRuntime->convertFromDevice(srcTensor, (const Tensor*)&interTensor, data_format, false);
    mOpenCLRuntime->printEventTime();

#ifdef ENABLE_OPENCL_TIME_PROFILER
    mOpenCLRuntime->commandQueue().finish();
    {
        AUTOTIME;
        mOpenCLRuntime->commandQueue().enqueueReadBuffer(*mHostBuffer.second, CL_TRUE, 0, needSize, hostPtr);
    }
#else
    mOpenCLRuntime->commandQueue().enqueueReadBuffer(*mHostBuffer.second, CL_TRUE, 0, needSize, hostPtr);
#endif

    if(dstTensor->getType().code == halide_type_int) {
        if(dstTensor->getType().bits == 8){
            tmpPtr = dstTensor->host<int8_t>();
            for(int i=0; i<needSize/4; i++) {
                ((int8_t*)tmpPtr)[i] = (int8_t)((float*)hostPtr)[i];
            }
        } else if(dstTensor->getType().bits == 32){
            tmpPtr = dstTensor->host<int32_t>();
            for(int i=0; i<needSize/4; i++) {
                ((int32_t*)tmpPtr)[i] = (int32_t)((float*)hostPtr)[i];
            }
        }
        if(hostPtr != nullptr) {
            free(hostPtr);
            hostPtr = nullptr;
        }
    } else if(dstTensor->getType().code == halide_type_uint){
        if(dstTensor->getType().bits == 8){
            tmpPtr = dstTensor->host<uint8_t>();
            for(int i=0; i<needSize/4; i++) {
                ((uint8_t*)tmpPtr)[i] = (uint8_t)((float*)hostPtr)[i];
            }
        } else if(dstTensor->getType().bits == 32){
            tmpPtr = dstTensor->host<uint32_t>();
            for(int i=0; i<needSize/4; i++) {
                ((uint32_t*)tmpPtr)[i] = (uint32_t)((float*)hostPtr)[i];
            }
        }
        if(hostPtr != nullptr) {
            free(hostPtr);
            hostPtr = nullptr;
        }
    }
}


void CLRuntime::convertToDevice(const Tensor* srcTensor, const Tensor* dstTensor, MNN_DATA_FORMAT data_format, bool svmFlag) const {
    // Format: Host -> OpenCL
    #ifndef MNN_OPENCL_BUFFER_CLOSED
    if(mOpenCLRuntime->getGpuMemType() == BUFFER)
    {
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
        int cPack = TensorUtils::getTensorChannelPack(dstTensor);
        if (cPack == 16 && mOpenCLRuntime->isSupportedIntelSubgroup()) {
            if (MNN_DATA_FORMAT_NHWC == data_format) {
                OpenCL::converNCHWOrNHWCBufferToNC4HW4OrNC16HW16Buffer(srcTensor, const_cast<Tensor*>(dstTensor),
                                                *const_cast<cl::Kernel*>(&mNHWCBufferToNC16HW16BufferInp), "nhwc_buffer_to_nc16hw16_buffer", mOpenCLRuntime.get(), true, false, svmFlag);
            } else if (MNN_DATA_FORMAT_NCHW == data_format) {
                OpenCL::converNCHWOrNHWCBufferToNC4HW4OrNC16HW16Buffer(srcTensor, const_cast<Tensor*>(dstTensor),
                                                *const_cast<cl::Kernel*>(&mNCHWBufferToNC16HW16BufferInp), "nchw_buffer_to_nc16hw16_buffer", mOpenCLRuntime.get(), true, false, svmFlag);
            } else if (MNN_DATA_FORMAT_NC4HW4 == data_format) {
                OpenCL::convertNC4HW4BufferBetweenNC16HW16Buffer(srcTensor, const_cast<Tensor*>(dstTensor),
                                                *const_cast<cl::Kernel*>(&mNC4HW4BufferToNC16HW16BufferInp), "nc4hw4_buffer_to_nc16hw16_buffer", mOpenCLRuntime.get(), InpTrans, false, svmFlag, true, false);
            } else {
                MNN_PRINT("input data format not support or subgroup\n");
                MNN_ASSERT(false);
            }
        }else
#endif        
        {
            if (MNN_DATA_FORMAT_NHWC == data_format) {
                OpenCL::converNCHWOrNHWCBufferToNC4HW4OrNC16HW16Buffer(srcTensor, const_cast<Tensor*>(dstTensor),
                                                 *const_cast<cl::Kernel*>(&mNHWCBufferToNC4HW4BufferInp), "nhwc_buffer_to_nc4hw4_buffer",mOpenCLRuntime.get(), true, false, svmFlag);
            } else if (MNN_DATA_FORMAT_NCHW == data_format) {
                OpenCL::converNCHWOrNHWCBufferToNC4HW4OrNC16HW16Buffer(srcTensor, const_cast<Tensor*>(dstTensor),
                                                 *const_cast<cl::Kernel*>(&mNCHWBufferToNC4HW4BufferInp), "nchw_buffer_to_nc4hw4_buffer",mOpenCLRuntime.get(), true, false, svmFlag);
            } else if (MNN_DATA_FORMAT_NC4HW4 == data_format) {
                OpenCL::convertNC4HW4BufferToNC4HW4Buffer(srcTensor, const_cast<Tensor*>(dstTensor),
                                                 *const_cast<cl::Kernel*>(&mNC4HW4BufferToNC4HW4BufferInp), mOpenCLRuntime.get(), InpTrans, false, svmFlag, true, false);
            } else {
                MNN_PRINT("input data format not support\n");
                MNN_ASSERT(false);
            }
        }
    }
    else
    #endif /* MNN_OPENCL_BUFFER_CLOSED */
    {
        if (MNN_DATA_FORMAT_NHWC == data_format) {
            OpenCL::convertNHWCBufferToImage(srcTensor, const_cast<Tensor*>(dstTensor),
                                             *const_cast<cl::Kernel*>(&mNHWCBufferToImageFloat), mOpenCLRuntime.get(), false, svmFlag);
        } else if (MNN_DATA_FORMAT_NCHW == data_format) {
            OpenCL::convertNCHWBufferToImage(srcTensor, const_cast<Tensor*>(dstTensor),
                                             *const_cast<cl::Kernel*>(&mNCHWBufferToImageFloat), mOpenCLRuntime.get(), false, svmFlag);
        } else if (MNN_DATA_FORMAT_NC4HW4 == data_format) {
            OpenCL::convertNC4HW4BufferToImage(srcTensor, const_cast<Tensor*>(dstTensor),
                                               *const_cast<cl::Kernel*>(&mNC4HW4BufferToImageFloat),
                                               mOpenCLRuntime.get(), false, svmFlag);
        } else {
            MNN_PRINT("data format not support\n");
            MNN_ASSERT(false);
        }
    }
}


void OpenCLBackend::copyToDevice(const Tensor* srcTensor, const Tensor* dstTensor) const{
    auto needSize = srcTensor->size();

    void* hostPtr;
    void* tmpPtr;
    if(srcTensor->getType().code == halide_type_int) {
        //Copy maybe slow, TODO
        if(srcTensor->getType().bits == 8){
            tmpPtr = srcTensor->host<int8_t>();
            needSize *= 4;
            hostPtr = malloc(needSize);
            for(int i=0; i<needSize/4; i++) {
                ((float*)hostPtr)[i] = (float)((int8_t*)tmpPtr)[i];
            }
        } else if(srcTensor->getType().bits == 32){
            tmpPtr = srcTensor->host<int32_t>();
            hostPtr = malloc(needSize);
            for(int i=0; i<needSize/4; i++) {
                ((float*)hostPtr)[i] = (float)((int32_t*)tmpPtr)[i];
            }
        }

    } else if(srcTensor->getType().code == halide_type_uint){
        //Copy maybe slow, TODO
        if(srcTensor->getType().bits == 8){
            tmpPtr = srcTensor->host<uint8_t>();
            needSize *= 4;
            hostPtr = malloc(needSize);
            for(int i=0; i<needSize/4; i++) {
                ((float*)hostPtr)[i] = (float)((uint8_t*)tmpPtr)[i];
            }
        } else if(srcTensor->getType().bits == 32){
            tmpPtr = srcTensor->host<uint32_t>();
            hostPtr = malloc(needSize);
            for(int i=0; i<needSize/4; i++) {
                ((float*)hostPtr)[i] = (float)((uint32_t*)tmpPtr)[i];
            }
        }
    } else {
        hostPtr                = srcTensor->host<float>();
    }

    _allocHostBuffer(needSize);

    MNN::Tensor interTensor(srcTensor, srcTensor->getDimensionType(), false);
    interTensor.buffer().device = (uint64_t)mHostBuffer.second.get();

    #ifdef ENABLE_OPENCL_TIME_PROFILER
    mOpenCLRuntime->commandQueue().finish();
    {
        AUTOTIME;
        mOpenCLRuntime->commandQueue().enqueueWriteBuffer(*mHostBuffer.second, CL_TRUE, 0, srcTensor->elementSize()*sizeof(float), hostPtr);
    }
    #else
    mOpenCLRuntime->commandQueue().enqueueWriteBuffer(*mHostBuffer.second, CL_TRUE, 0, srcTensor->elementSize()*sizeof(float), hostPtr);
    #endif

    //Covert format
    MNN_DATA_FORMAT data_format = TensorUtils::getDescribe(srcTensor)->dimensionFormat;
    mCLRuntime->convertToDevice((const Tensor*)&interTensor, dstTensor, data_format, false);

    if(srcTensor->getType().code == halide_type_uint || srcTensor->getType().code == halide_type_int){
        mOpenCLRuntime.get()->commandQueue().finish();
        if(nullptr != hostPtr){
            free(hostPtr);
            hostPtr = nullptr;
        }
    }
    return;
}

void CLRuntime::copyBetweenDevice(const Tensor* srcTensor, const Tensor* dstTensor) const{
    #ifndef MNN_OPENCL_BUFFER_CLOSED
    if(mOpenCLRuntime->getGpuMemType() == BUFFER)
    {
        OpenCL::convertNC4HW4BufferToNC4HW4Buffer(srcTensor, const_cast<Tensor*>(dstTensor),
                                         *const_cast<cl::Kernel*>(&mNC4HW4BufferToNC4HW4Buffer), mOpenCLRuntime.get(), NoTrans);
    }
    else
    #endif /* MNN_OPENCL_BUFFER_CLOSED */
    {
        std::vector<int> bufferShape = MNN::OpenCL::tensorShapeFormat(srcTensor);

        mOpenCLRuntime.get()->commandQueue().enqueueCopyImage(
                openCLImage(srcTensor), openCLImage(dstTensor),
                {0, 0, 0}, {0, 0, 0},
                {(size_t)bufferShape[2]* UP_DIV(bufferShape[3], 4), (size_t)bufferShape[0]*bufferShape[1], 1});
    }
    return;
}


void OpenCLBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start onCopyBuffer !\n");
#endif
    clearRecord();
    auto srcDevice = (srcTensor->deviceId() != 0 && srcTensor->deviceId() != 1);
    auto dstDevice = (dstTensor->deviceId() != 0 && dstTensor->deviceId() != 1);
    if (!srcDevice && dstDevice) {
        copyToDevice(srcTensor, dstTensor);
    }else if(srcDevice && !dstDevice){
        copyFromDevice(srcTensor, dstTensor);
    }else if(srcDevice && dstDevice){
        mCLRuntime->copyBetweenDevice(srcTensor, dstTensor);
    }else{
        MNN_PRINT("onCopyBuffer float error !!! \n");
    }

#ifdef LOG_VERBOSE
    MNN_PRINT("end onCopyBuffer !\n");
#endif
}

void* OpenCLBackend::allocMapTensorMemory(int length, bool svmFlag, cl_device_svm_capabilities svm_cap_) {
    if(length <= mMapMem.first) {
        return mMapMem.second;
    }

#ifdef MNN_OPENCL_SVM_ENABLE
    if(svmFlag)
    {
        if(mMapMem.first != 0) {
            //Release small SVM Memory
            clSVMFree(mOpenCLRuntime->context().get(), mMapMem.second);
        }
        //Alloc proper SVM Memory
        cl_svm_mem_flags flags = CL_MEM_READ_WRITE;
        flags |= (svm_cap_ & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) ? CL_MEM_SVM_FINE_GRAIN_BUFFER : 0;
        flags |= ((svm_cap_ & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) && (svm_cap_ & CL_DEVICE_SVM_ATOMICS)) ? CL_MEM_SVM_ATOMICS : 0;


        mMapMem.second = clSVMAlloc(mOpenCLRuntime->context().get(), flags, length, 0);
        if(mMapMem.second == nullptr) {
            MNN_PRINT("SVM Alloc Failed\n");
        }
    }
    else
#endif
    {
        if(mMapMem.first != 0) {
            free(mMapMem.second);
            mMapMem.second = nullptr;
        }
        mMapMem.second = malloc(length);
    }
    mMapMem.first = length;
    return mMapMem.second;

}

void* OpenCLBackend::onMapTensor(Tensor::MapType mtype, Tensor::DimensionType dtype, const Tensor* srcTensor) {
    auto needSize = srcTensor->size();
    clearRecord();
#ifdef MNN_OPENCL_SVM_ENABLE
    auto svm_cap_ = mOpenCLRuntime->getSvmCapabilities();
    bool use_svm = (svm_cap_ & CL_DEVICE_SVM_FINE_GRAIN_BUFFER);//support fine grain svm
    use_svm |= ((svm_cap_ & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) && mOpenCLRuntime->getGpuType() == ADRENO);//support coarse grain svm and adreno gpu

    mUseSvm = (mOpenCLRuntime->getCLVersion() > 1.99f && use_svm);
    if(mUseSvm) {// CL version beyond 2.0 & support svm
        svmPtr = allocMapTensorMemory(needSize, true, svm_cap_);

        if(mtype == Tensor::MAP_TENSOR_READ) {
            //tmpTensor alloc
            MNN::Tensor tmpTensor(srcTensor, dtype, false);
            tmpTensor.buffer().device = (uint64_t)svmPtr;

            //Convert format
            MNN_DATA_FORMAT format_type = MNN_DATA_FORMAT_NCHW;
            if(dtype == MNN::Tensor::TENSORFLOW) {
                format_type = MNN_DATA_FORMAT_NHWC;
            } else if(dtype == MNN::Tensor::CAFFE_C4) {
                format_type = MNN_DATA_FORMAT_NC4HW4;
            }
            mCLRuntime->convertFromDevice(srcTensor, &tmpTensor, format_type, true);
        }

        if(svm_cap_ & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) {
            //Make sure command finished
            mOpenCLRuntime->commandQueue().finish();
            return svmPtr;
        }

        auto map_flag = CL_MAP_WRITE;
        if(mtype == Tensor::MAP_TENSOR_READ) {
            map_flag = CL_MAP_READ;
        }

        cl_int res = clEnqueueSVMMap(mOpenCLRuntime->commandQueue().get(), true, map_flag, svmPtr, needSize, 0, nullptr, nullptr);

        MNN_CHECK_CL_SUCCESS(res, "svm_map")
        return svmPtr;
    }
#endif

    /**
    Not Support Svm, Use onopyBuffer
     */
    svmPtr = allocMapTensorMemory(needSize, false);

    if(mtype == Tensor::MAP_TENSOR_READ) {
        //tmpTensor alloc
        MNN::Tensor tmpTensor(srcTensor, dtype, false);
        tmpTensor.buffer().host = (uint8_t *)svmPtr;

        //use onCopyBuffer
        onCopyBuffer(srcTensor, &tmpTensor);
    }
    return svmPtr;
}

bool OpenCLBackend::onUnmapTensor(Tensor::MapType mtype, Tensor::DimensionType dtype, const Tensor* dstTensor, void* mapPtr) {
#ifdef MNN_OPENCL_SVM_ENABLE
    auto svm_cap_ = mOpenCLRuntime->getSvmCapabilities();
    if(mUseSvm) {// CL version beyond 2.0 & support svm

        //If COARSE_SVM, Unmap first
        if(!(svm_cap_ & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)) {
            cl_int res = clEnqueueSVMUnmap(mOpenCLRuntime->commandQueue().get(), svmPtr, 0, nullptr, nullptr);
            MNN_CHECK_CL_SUCCESS(res, "svm_unmap")
        }

        if(mtype == Tensor::MAP_TENSOR_WRITE) {
            //interTensor alloc
            MNN::Tensor interTensor(dstTensor, dtype, false);
            interTensor.buffer().device = (uint64_t)svmPtr;

            //Convert format
            MNN_DATA_FORMAT format_type = MNN_DATA_FORMAT_NCHW;
            if(dtype == MNN::Tensor::TENSORFLOW) {
                format_type = MNN_DATA_FORMAT_NHWC;
            } else if(dtype == MNN::Tensor::CAFFE_C4) {
                format_type = MNN_DATA_FORMAT_NC4HW4;
            }
            mCLRuntime->convertToDevice(&interTensor, dstTensor, format_type, true);
        }
        mOpenCLRuntime->commandQueue().finish();

        return true;
    }
#endif

    /**
    Not Support Svm, Use onopyBuffer
     */
    if(mtype == Tensor::MAP_TENSOR_WRITE) {
        //srcTensor alloc
        MNN::Tensor srcTensor(dstTensor, dtype, false);
        srcTensor.buffer().host = (uint8_t *)svmPtr;

        //use onCopyBuffer
        onCopyBuffer(&srcTensor, dstTensor);
    }
    return true;
}

bool OpenCLBackend::addCreator(std::pair<OpType, GpuMemObject> t, Creator* c) {
    auto map = gCreator();
    if (map->find(t) != map->end()) {
        MNN_PRINT("Error: %d type, %d GpuMemObject has be added\n", t.first, t.second);
        return false;
    }
    map->insert(std::make_pair(t, c));
    return true;
}

// -----------------------------------------------------------------------------
// Runtime Register
// -----------------------------------------------------------------------------
class CLRuntimeCreator : public RuntimeCreator {
    virtual Runtime* onCreate(const Backend::Info& info) const {
    #ifdef MNN_USE_LIB_WRAPPER
        OpenCLSymbolsOperator::createOpenCLSymbolsOperatorSingleInstance();
        if (nullptr == OpenCLSymbolsOperator::getOpenclSymbolsPtr()) {
            MNN_PRINT("OpenCL init error, fallback ... \n");
            return nullptr;
        }
        if (true == OpenCLSymbolsOperator::getOpenclSymbolsPtr()->isError()) {
            MNN_PRINT("Parsing OpenCL symbols error !!! \n");
            return nullptr;
        }
    #endif
        int platform_id = 0;
        int device_id = 0;
        int platform_size = 0;
        if (nullptr != info.user) {
            if (info.user->sharedContext != nullptr) {
                platform_id   = ((MNNDeviceContext*)info.user->sharedContext)->platformId;
                device_id     = ((MNNDeviceContext*)info.user->sharedContext)->deviceId;
                platform_size = ((MNNDeviceContext*)info.user->sharedContext)->platformSize;
            }
        }
        auto rt = new CLRuntime(info, platform_size, platform_id, device_id);
        if(rt->isCLRuntimeError() == true) {
            delete rt;
            return nullptr;
        }
        return rt;
    }
    virtual bool onValid(Backend::Info& info) const {
        return true;
    }
};

DataType OpenCLBackend::getDataType(const Tensor* tensor) {
    auto des = TensorUtils::getDescribe(tensor);
    if (nullptr == des->quantAttr.get()) {
        return DataType_DT_FLOAT;
    }
    return des->type;
}

cl_channel_type OpenCLBackend::fpType() {
    if (getOpenCLRuntime()->isSupportedFP16() &&
        mPrecision != BackendConfig::Precision_High) {
        return CL_HALF_FLOAT;
    }
    return CL_FLOAT;
}

int OpenCLBackend::fpBytes() {
    return (fpType() == CL_FLOAT ?  sizeof(float) : sizeof(half_float::half));
}

void OpenCLBackend::clearRecord() const{
#if !defined(ENABLE_OPENCL_TIME_PROFILER) && defined(MNN_USE_LIB_WRAPPER)
    if(mUseRecordQueue && mDevideOpRecord){
        for(int i = 0; i < mRecordings.size(); ++i){
            cl_int res = mOpenCLRuntime->commandQueue().EnqueueRecordingQCOM(mRecordings[i], 0, nullptr, 0, nullptr,
                  0, nullptr, 0, nullptr, 0, nullptr, nullptr);
            MNN_CHECK_CL_SUCCESS(res, "EnqueueRecordingQCOM");
        }
        mOpenCLRuntime->commandQueue().finish();
        mRecordings.clear();
    }
#endif
}

void OpenCLBackend::enqeueRecord() const{
#if !defined(ENABLE_OPENCL_TIME_PROFILER) && defined(MNN_USE_LIB_WRAPPER)
    if(mUseRecordQueue && !mDevideOpRecord){
        for(int i = 0; i < mRecordings.size(); ++i){
            cl_int res = mOpenCLRuntime->commandQueue().EnqueueRecordingQCOM(mRecordings[i], 0, nullptr, 0, nullptr,
                  0, nullptr, 0, nullptr, 0, nullptr, nullptr);
            MNN_CHECK_CL_SUCCESS(res, "EnqueueRecordingQCOM");
        }
        mOpenCLRuntime->commandQueue().finish();
    }
#endif
}

void OpenCLBackend::releaseRecord(){
#if !defined(ENABLE_OPENCL_TIME_PROFILER) && defined(MNN_USE_LIB_WRAPPER)
    if(mUseRecordQueue  && !mDevideOpRecord){
        for(int i = 0; i < mRecordings.size(); ++i){
            cl_int res = clReleaseRecordingQCOM(mRecordings[i]);
            MNN_CHECK_CL_SUCCESS(res, "clReleaseRecordingQCOM");
        }
        mRecordings.clear();
    }
#endif
}

void OpenCLBackend::startRecord(cl_recording_qcom &recording){
#if !defined(ENABLE_OPENCL_TIME_PROFILER) && defined(MNN_USE_LIB_WRAPPER)
    if(!mUseRecordQueue){
        return;
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("start startRecord !\n");
#endif
    cl_int res = CL_SUCCESS;
    if(mDevideOpRecord){
        if(recording != NULL){
            clReleaseRecordingQCOM(recording);
        }
        recording = mOpenCLRuntime->recordableQueue().NewRecordingQCOM(&res);
        MNN_CHECK_CL_SUCCESS(res, "clNewRecordingQCOM");
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("end startRecord !\n");
#endif
#endif //ENABLE_OPENCL_TIME_PROFILER
}

void OpenCLBackend::endRecord(cl_recording_qcom &recording, bool flag){
#if !defined(ENABLE_OPENCL_TIME_PROFILER) && defined(MNN_USE_LIB_WRAPPER)
    if(!mUseRecordQueue){
        return;
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("start endRecord !\n");
#endif
    if(mDevideOpRecord){
        cl_int res = CL_SUCCESS;
        res = clEndRecordingQCOM(recording);
        MNN_CHECK_CL_SUCCESS(res, "clEndRecordingQCOM");
    } else if(flag) {
        if(!mRecordings.empty()){
            cl_int res = clEndRecordingQCOM(mRecordings.back());
            mRecordNums = 0;
            MNN_CHECK_CL_SUCCESS(res, "clEndRecordingQCOM");
        }
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("end endRecord !\n");
#endif
#endif //ENABLE_OPENCL_TIME_PROFILER
}

void OpenCLBackend::recordKernel2d(const ::cl::Kernel &kernel, const std::vector<uint32_t> &gws, const std::vector<uint32_t> &lws) {
#if !defined(ENABLE_OPENCL_TIME_PROFILER) && defined(MNN_USE_LIB_WRAPPER)
    if(!mUseRecordQueue){
        return;
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("start record2dKernel !\n");
#endif
    cl_int res = CL_SUCCESS;
    if(!mDevideOpRecord){
        if(mRecordNums == 0){
            cl_recording_qcom recording = mOpenCLRuntime->recordableQueue().NewRecordingQCOM(&res);
            MNN_CHECK_CL_SUCCESS(res, "clNewRecordingQCOM");
            mRecordings.emplace_back(recording);
        }else if(mRecordNums == mUseRecordableQueueSize){
            res = clEndRecordingQCOM(mRecordings.back());
            MNN_CHECK_CL_SUCCESS(res, "clEndRecordingQCOM");
            cl_recording_qcom recording = mOpenCLRuntime->recordableQueue().NewRecordingQCOM(&res);
            MNN_CHECK_CL_SUCCESS(res, "clNewRecordingQCOM");
            mRecordings.emplace_back(recording);
            mRecordNums = 0;
        }
        mRecordNums++;
    }
    
    std::vector<uint32_t> internalGlobalWS = gws;
    for (size_t i = 0; i < 2; ++i) {
        internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
    }

    if(lws[0]==0 || lws[1]==0){
        res = mOpenCLRuntime->recordableQueue().enqueueNDRangeKernel(
            kernel, cl::NullRange, cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]), cl::NullRange, nullptr, nullptr);

    }else{
        res = mOpenCLRuntime->recordableQueue().enqueueNDRangeKernel(
            kernel, cl::NullRange, cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]), cl::NDRange(lws[0], lws[1]), nullptr, nullptr);
    }
    MNN_CHECK_CL_SUCCESS(res, "recordKernel2d");

#ifdef LOG_VERBOSE
    MNN_PRINT("end record2dKernel !\n");
#endif
#endif //ENABLE_OPENCL_TIME_PROFILER
}

void OpenCLBackend::recordKernel3d(const ::cl::Kernel &kernel, const std::vector<uint32_t> &gws, const std::vector<uint32_t> &lws) {
#if !defined(ENABLE_OPENCL_TIME_PROFILER) && defined(MNN_USE_LIB_WRAPPER)
    if(!mUseRecordQueue){
        return;
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("start record3dKernel !\n");
#endif
    cl_int res = CL_SUCCESS;
    std::vector<uint32_t> internalGlobalWS = gws;
    for (size_t i = 0; i < 3; ++i) {
        internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
    }
    if(!mDevideOpRecord){
        if(mRecordNums == 0){
            cl_recording_qcom recording = mOpenCLRuntime->recordableQueue().NewRecordingQCOM(&res);
            MNN_CHECK_CL_SUCCESS(res, "clNewRecordingQCOM");
            mRecordings.emplace_back(recording);
        }else if(mRecordNums == mUseRecordableQueueSize){
            res = clEndRecordingQCOM(mRecordings.back());
            MNN_CHECK_CL_SUCCESS(res, "clEndRecordingQCOM");
            cl_recording_qcom recording = mOpenCLRuntime->recordableQueue().NewRecordingQCOM(&res);
            MNN_CHECK_CL_SUCCESS(res, "clNewRecordingQCOM");
            mRecordings.emplace_back(recording);
            mRecordNums = 0;
        }
        mRecordNums++;
    }

    if(lws[0]==0 || lws[1]==0 || lws[2]==0){
        res = mOpenCLRuntime->recordableQueue().enqueueNDRangeKernel(
            kernel, cl::NullRange, cl::NDRange(internalGlobalWS[0], internalGlobalWS[1], internalGlobalWS[2]), cl::NullRange, nullptr, nullptr);

    }else{
        res = mOpenCLRuntime->recordableQueue().enqueueNDRangeKernel(
            kernel, cl::NullRange, cl::NDRange(internalGlobalWS[0], internalGlobalWS[1], internalGlobalWS[2]), cl::NDRange(lws[0], lws[1], lws[2]), nullptr, nullptr);
    }
    MNN_CHECK_CL_SUCCESS(res, "recordKernel3d");
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end record3dKernel !\n");
#endif
#endif //ENABLE_OPENCL_TIME_PROFILER
}

#ifdef MNN_OPENCL_SEP_BUILD
bool placeholder = []() {
    static std::once_flag createOnce;
    std::call_once(createOnce, []() {
        MNNInsertExtraRuntimeCreator(MNN_FORWARD_OPENCL, new CLRuntimeCreator, false);
    });
    return true;
}();
#else
void registerOpenCLRuntimeCreator() {
    registerOpenCLOps();
    MNNInsertExtraRuntimeCreator(MNN_FORWARD_OPENCL, new CLRuntimeCreator, false);
}
#endif
} // namespace OpenCL

} // namespace MNN