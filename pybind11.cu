#include <iostream>
//#include <fstream>
#include <nvjpeg.h>
//#include <cuda_runtime.h>
#include <string>
#include <filesystem>
#include <algorithm>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
//#include <format>
//#include "nameof.hpp"

#define CHECK_CUDA(call) {cudaError_t _e = (call); if (_e != cudaSuccess){throw runtime_error("CUDA Runtime error at "s+__FILE__+":"s+to_string(__LINE__)+" as #"s+to_string(static_cast<int>(_e)));}}
#define CHECK_NVJPEG(call) {nvjpegStatus_t _e = (call);if (_e != NVJPEG_STATUS_SUCCESS){throw runtime_error("NVJPEG error at "s+__FILE__+":"s+to_string(__LINE__)+" as #"s+to_string(static_cast<int>(_e)));}}

using namespace std;
namespace fs = filesystem;
//string source_dir = R"(D:\helloproject-ai-data\blog_images)";

pair<vector<float>, pair<int, int>> decode_image(const pybind11::bytes &file, const string &normalize);
//void infer(Ort::Session session, Ort::MemoryInfo memInfo, const vector<char> &data, int width, int height);

//int main() {
////    nvjpegStatus_t status;
//    vector<fs::directory_entry> member_names;
//    for (auto &person_name: fs::directory_iterator(source_dir)) {
//        for (auto &file_name: fs::directory_iterator(person_name)) {
////            cout << file_name.path().string() << endl;
//            member_names.push_back(file_name);
//        }
//    }
//    for (const auto &file: member_names) {
//        ifstream ifs(file.path(), ios::binary);
////        ifs.seekg(0, ios::end);
////        auto file_length = ifs.tellg();
////        ifs.seekg(0, ios::beg);
////    istream_iterator<unsigned char> begin(ifs), end;
//        vector<unsigned char> file_content(
//                (istreambuf_iterator<char>(ifs)),
//                istreambuf_iterator<char>()
//        );
//
//        ifs.close();
//        auto raw_img = decode_image(file_content, "none");
////        cudaFree(raw_img.first);
//    }
//    return 0;
//}
//
//int counter = 0;

struct none_char_to_float {
    __host__ __device__ float operator()(unsigned char x) const {
        return static_cast<float>(x) / 255.0f;
    }
};

struct imagenet_r_char_to_float {
    __host__ __device__ float operator()(unsigned char x) const {
        return (static_cast<float>(x) - 0.485F * 255.0F) / 0.229F * 255.0F;
    }
};

struct imagenet_g_char_to_float {
    __host__ __device__ float operator()(unsigned char x) const {
        return (static_cast<float>(x) - 0.456F * 255.0F) / 0.224F * 255.0F;
    }
};

struct imagenet_b_char_to_float {
    __host__ __device__ float operator()(unsigned char x) const {
        return (static_cast<float>(x) - 0.406F * 255.0F) / 0.225F * 255.0F;
    }
};

bool nvInit = false;
cudaStream_t stream = nullptr;
nvjpegJpegState_t state = nullptr;
nvjpegHandle_t handle = nullptr;

pair<vector<float>, pair<int, int>> decode_image(const pybind11::bytes &file, const string &normalize) {
    if (!nvInit) {
        CHECK_NVJPEG(nvjpegCreate(nvjpegBackend_t::NVJPEG_BACKEND_GPU_HYBRID, nullptr, &handle))
        CHECK_NVJPEG(nvjpegJpegStateCreate(handle, &state))
        CHECK_CUDA(cudaStreamCreate(&stream))
        nvInit = true;
    }
    string a = file;
//    cout << a << endl;
    vector<unsigned char> file_data(a.begin(), a.end());
//    nvjpegStatus_t status;
//    cout << "copied to vector" << endl;
//    cudaEvent_t startEvent = nullptr, stopEvent = nullptr;
//    CHECK_CUDA(cudaEventCreate(&startEvent, cudaEventBlockingSync))
//    CHECK_CUDA(cudaEventCreate(&stopEvent, cudaEventBlockingSync))
    int n_components;
    int widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
    nvjpegChromaSubsampling_t subsampling_t;
//    char *data = file_content.data();
    CHECK_NVJPEG(nvjpegGetImageInfo(handle, reinterpret_cast<const unsigned char *>(file_data.data()), file_data.size(),
                                    &n_components, &subsampling_t, widths, heights))
    nvjpegImage_t image;
    for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
        image.pitch[c] = widths[0];
        cudaMalloc((void **) &image.channel[c], widths[0] * heights[0]);
//        image.channel[c] = static_cast<unsigned char *>(malloc(widths[0] * heights[0]));
    }
    CHECK_NVJPEG(
            nvjpegDecode(handle, state, reinterpret_cast<const unsigned char *>(file_data.data()), file_data.size(),
                         nvjpegOutputFormat_t::NVJPEG_OUTPUT_RGB, &image, stream))
//    FILE *test;
//    fopen_s(&test, "test_file.raw", "wb");
//    vector<char> output(widths[0] * heights[0] * 3);
    thrust::device_vector<float> dest_gpu(3 * widths[0] * heights[0]);
    if (normalize == "imagenet") {
        thrust::transform(thrust::device, image.channel[0], image.channel[0] + 0 * widths[0] * heights[0],
                          dest_gpu.begin() + widths[0] * heights[0] * 0, imagenet_r_char_to_float());
        thrust::transform(thrust::device, image.channel[1], image.channel[1] + 1 * widths[0] * heights[0],
                          dest_gpu.begin() + widths[0] * heights[0] * 1, imagenet_g_char_to_float());
        thrust::transform(thrust::device, image.channel[2], image.channel[2] + 2 * widths[0] * heights[0],
                          dest_gpu.begin() + widths[0] * heights[0] * 2, imagenet_b_char_to_float());
    } else if (normalize == "none") {
        for (int i = 0; i < 3; i++) {
            thrust::transform(thrust::device, image.channel[i], image.channel[i] + widths[0] * heights[0],
                              dest_gpu.begin() + widths[0] * heights[0] * i, none_char_to_float());
        }
    } else {
        throw runtime_error("normalize must be 'imagenet' or 'none'");
    }

    for (auto &i: image.channel) {
        cudaFree(i);
    }


//    fclose(test);
//    cout << counter++ << " : " << file_name.path().string() << endl;
//    cout << "width:" << widths[0] << endl;
//    cout << "height:" << heights[0] << endl;
//    cout << image.pitch << endl;
    vector<float> dest(3 * widths[0] * heights[0]);
    thrust::copy(dest_gpu.begin(), dest_gpu.end(), dest.begin());
    return {dest, {*widths, *heights}};
}

//void infer(Ort::Session session, Ort::MemoryInfo memInfo, const vector<char> &data, int width, int height) {
//    const int64_t input_shape[4] = {1, 3, height, width};
//    auto input_tensor = Ort::Value::CreateTensor<char>(memInfo, const_cast<char *>(data.data()), data.size(),
//                                                       input_shape, 4);
//}

PYBIND11_MODULE(nvjpeg_decoder, m) {
    m.def("decode", &decode_image, "pass image binary and return decoded numpy array.");
}