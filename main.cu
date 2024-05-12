#include <iostream>
#include <fstream>
#include <nvjpeg.h>
#include <cuda_runtime.h>
#include <string>
#include <filesystem>
#include <algorithm>
#include <onnxruntime/tensorrt_provider_options.h>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

#define CHECK_CUDA(call) {cudaError_t _e = (call);if (_e != cudaSuccess){std::cout << "CUDA Runtime failure: '#" << _e << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl;exit(1);}}
#define CHECK_NVJPEG(call) {nvjpegStatus_t _e = (call);if (_e != NVJPEG_STATUS_SUCCESS){cout << "NVJPEG failure: '#" << _e << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl;exit(1);                                                            }                                                                       }

using namespace std;
namespace fs = filesystem;
string source_dir = R"(D:\helloproject-ai-data\blog_images)";

pair<float *, pair<int, int>>
decode_image(const fs::directory_entry &file_name, nvjpegHandle_t handle, nvjpegJpegState_t state, cudaStream_t stream);

//void infer(Ort::Session session, Ort::MemoryInfo memInfo, const vector<char> &data, int width, int height);

int main() {
    vector<fs::directory_entry> member_names;
    for (auto &person_name: fs::directory_iterator(source_dir)) {
        for (auto &file_name: fs::directory_iterator(person_name)) {
//            cout << file_name.path().string() << endl;
            member_names.push_back(file_name);
        }
    }
    nvjpegHandle_t handle = nullptr;
    CHECK_NVJPEG(nvjpegCreate(nvjpegBackend_t::NVJPEG_BACKEND_GPU_HYBRID, nullptr, &handle));
    nvjpegJpegState_t state = nullptr;
    CHECK_NVJPEG(nvjpegJpegStateCreate(handle, &state));
    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));
//    std::copy(member_names.begin(), member_names.end(), ostream_iterator<fs::directory_entry>(cout, "\n"));
    Ort::Env env{OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR};
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    int device_id = 0;
    const auto &api = Ort::GetApi();
    OrtTensorRTProviderOptionsV2 *TensorRtProviderOptionsV2;
    Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&TensorRtProviderOptionsV2));
//    unique_ptr<OrtTensorRTProviderOptionsV2 , decltype(api.ReleaseTensorRTProviderOptions)> rel_TensorRTOptions(TensorRtProviderOptionsV2,api.ReleaseTensorRTProviderOptions);
    Ort::ThrowOnError(api.UpdateTensorRTProviderOptionsWithValue(TensorRtProviderOptionsV2, "device_id", (void *) "0"));
    Ort::ThrowOnError(
            api.UpdateTensorRTProviderOptionsWithValue(TensorRtProviderOptionsV2, "trt_fp16_enable", (void *) "1"));
    Ort::ThrowOnError(
            api.UpdateTensorRTProviderOptionsWithValue(TensorRtProviderOptionsV2, "trt_fp8_enable", (void *) "1"));
    Ort::ThrowOnError(api.UpdateTensorRTProviderOptionsWithValue(TensorRtProviderOptionsV2, "trt_engine_cache_enable",
                                                                 (void *) "1"));
    Ort::ThrowOnError(api.UpdateTensorRTProviderOptionsWithValue(TensorRtProviderOptionsV2, "trt_engine_cache_path",
                                                                 (void *) "cache"));
    Ort::ThrowOnError(api.UpdateTensorRTProviderOptionsWithValue(TensorRtProviderOptionsV2, "trt_engine_timing_enable",
                                                                 (void *) "1"));
    Ort::ThrowOnError(api.UpdateTensorRTProviderOptionsWithValue(TensorRtProviderOptionsV2, "trt_engine_timing_path",
                                                                 (void *) "timing"));
    Ort::ThrowOnError(api.UpdateTensorRTProviderOptionsWithValue(TensorRtProviderOptionsV2, "user_compute_stream",
                                                                 (void *) stream));
    Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(static_cast<OrtSessionOptions *>(options),
                                                                            TensorRtProviderOptionsV2));
//    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(options, device_id));
    OrtCUDAProviderOptionsV2 *CudaProviderOptionsV2;
    Ort::ThrowOnError(api.CreateCUDAProviderOptions(&CudaProviderOptionsV2));
    Ort::ThrowOnError(api.UpdateCUDAProviderOptionsWithValue(CudaProviderOptionsV2, "device_id", (void *) "0"));
    Ort::ThrowOnError(
            api.UpdateCUDAProviderOptionsWithValue(CudaProviderOptionsV2, "user_compute_stream", (void *) stream));
    options.AppendExecutionProvider_CUDA_V2(*CudaProviderOptionsV2);

    Ort::Session session(env, L"resnet_retinaface.onnx", options);
    Ort::MemoryInfo memInfo(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
    for (const auto &file: member_names) {
        auto raw_img = decode_image(file, handle, state, stream);
        cudaFree(raw_img.first);
    }
    return 0;
}

int counter = 0;

struct none_char_to_float {
    __host__ __device__ float operator()(unsigned char x) const {
        return static_cast<float>(x) / 255.0f;
    }
};

pair<float *, pair<int, int>>
decode_image(const fs::directory_entry &file_name, nvjpegHandle_t handle, nvjpegJpegState_t state,
             cudaStream_t stream) {
//    nvjpegStatus_t status;
    ifstream ifs(file_name.path(), ios::binary);
    ifs.seekg(0, ios::end);
    auto file_length = ifs.tellg();
    ifs.seekg(0, ios::beg);
//    istream_iterator<unsigned char> begin(ifs), end;
    char *file_content = new char[file_length];
    ifs.read(file_content, file_length);
    ifs.close();
//    cudaEvent_t startEvent = nullptr, stopEvent = nullptr;
//    CHECK_CUDA(cudaEventCreate(&startEvent, cudaEventBlockingSync))
//    CHECK_CUDA(cudaEventCreate(&stopEvent, cudaEventBlockingSync))
    int n_components;
    int widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
    nvjpegChromaSubsampling_t subsampling_t;
//    char *data = file_content.data();
    CHECK_NVJPEG(nvjpegGetImageInfo(handle, reinterpret_cast<const unsigned char *>(file_content), file_length,
                                    &n_components, &subsampling_t, widths, heights));
    nvjpegImage_t image;
    for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
        image.pitch[c] = widths[0];
        cudaMalloc((void **) &image.channel[c], widths[0] * heights[0]);
//        image.channel[c] = static_cast<unsigned char *>(malloc(widths[0] * heights[0]));
    }
    CHECK_NVJPEG(nvjpegDecode(handle, state, reinterpret_cast<const unsigned char *>(file_content), file_length,
                              nvjpegOutputFormat_t::NVJPEG_OUTPUT_RGB, &image, stream));
    delete[] file_content;
//    FILE *test;
//    fopen_s(&test, "test_file.raw", "wb");
//    vector<char> output(widths[0] * heights[0] * 3);
    unsigned char *nvjpeg_recv;
    cudaMalloc((void **) &nvjpeg_recv, sizeof(unsigned char) * widths[0] * heights[0] * 3);
    for (int i = 0; i < 3; i++) {
//        auto recv = static_cast<unsigned char *>(malloc(widths[0] * heights[0]));
//        cudaMemcpy(recv, image.channel[i], widths[0] * heights[0], cudaMemcpyKind::cudaMemcpyDeviceToHost);
        cudaMemcpy(nvjpeg_recv + i * widths[0] * heights[0], image.channel[i], widths[0] * heights[0],
                   cudaMemcpyKind::cudaMemcpyDeviceToDevice);
//        copy(recv, recv + widths[0] * heights[0], back_inserter(output));
//        fwrite(recv, 1, widths[0] * heights[0], test);
        cudaFree(image.channel[i]);
//        free(recv);
    }
    cudaFree(image.channel[3]);
    float *trans_dest;
    cudaMalloc((void **) &trans_dest, sizeof(float) * widths[0] * heights[0] * 3);
    thrust::transform(thrust::device, nvjpeg_recv, nvjpeg_recv + 3 * widths[0] * heights[0], trans_dest,
                      none_char_to_float());
    cudaFree(nvjpeg_recv);

//    fclose(test);
    cout << counter++ << " : " << file_name.path().string() << endl;
//    cout << "width:" << widths[0] << endl;
//    cout << "height:" << heights[0] << endl;
//    cout << image.pitch << endl;
    return {trans_dest, {*widths, *heights}};
}

//void infer(Ort::Session session, Ort::MemoryInfo memInfo, const vector<char> &data, int width, int height) {
//    const int64_t input_shape[4] = {1, 3, height, width};
//    auto input_tensor = Ort::Value::CreateTensor<char>(memInfo, const_cast<char *>(data.data()), data.size(),
//                                                       input_shape, 4);
//}