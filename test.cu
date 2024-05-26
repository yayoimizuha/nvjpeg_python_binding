//
// Created by tomokazu on 24/05/26.
//
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
#include <nppi_geometry_transforms.h>
#include <nppi_data_exchange_and_initialization.h>
#include <nppdefs.h>
#include <fstream>

#define CHECK_CUDA(call) {cudaError_t _e = (call); if (_e != cudaSuccess){throw runtime_error("CUDA Runtime error at "s+__FILE__+":"s+to_string(__LINE__)+" as #"s+to_string(static_cast<int>(_e)));}}
#define CHECK_NVJPEG(call) {nvjpegStatus_t _e = (call);if (_e != NVJPEG_STATUS_SUCCESS){throw runtime_error("NVJPEG error at "s+__FILE__+":"s+to_string(__LINE__)+" as #"s+to_string(static_cast<int>(_e)));}}
#define CHECK_NPP(call) {NppStatus _e = (call);if (_e != NppStatus::NPP_SUCCESS){throw runtime_error("NPP error at "s+__FILE__+":"s+to_string(__LINE__)+" as #"s+to_string(static_cast<int>(_e)));}};


using namespace std;
namespace fs = filesystem;

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

pair<vector<float>, pair<double, pair<int, int>>>
decode_image(const vector<unsigned char> &file_data, const string &normalize, pair<int, int> dest_size) {
    if (!nvInit) {
        CHECK_NVJPEG(nvjpegCreate(nvjpegBackend_t::NVJPEG_BACKEND_GPU_HYBRID, nullptr, &handle))
        CHECK_NVJPEG(nvjpegJpegStateCreate(handle, &state))
        CHECK_CUDA(cudaStreamCreate(&stream))
        nvInit = true;
    }
    int n_components;
    int widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
    nvjpegChromaSubsampling_t subsampling_t;

    CHECK_NVJPEG(nvjpegGetImageInfo(handle, reinterpret_cast<const unsigned char *>(file_data.data()), file_data.size(),
                                    &n_components, &subsampling_t, widths, heights))
    nvjpegImage_t image;
    for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
        image.pitch[c] = widths[0];
        cudaMalloc((void **) &image.channel[c], widths[0] * heights[0]);

    }
    CHECK_NVJPEG(
            nvjpegDecode(handle, state, reinterpret_cast<const unsigned char *>(file_data.data()), file_data.size(),
                         nvjpegOutputFormat_t::NVJPEG_OUTPUT_RGB, &image, stream))

    thrust::device_vector<float> dest_gpu(3 * widths[0] * heights[0]);
    if (normalize == "imagenet") {
        thrust::transform(thrust::device, image.channel[0], image.channel[0] + widths[0] * heights[0],
                          dest_gpu.begin() + widths[0] * heights[0] * 0, imagenet_r_char_to_float());
        thrust::transform(thrust::device, image.channel[1], image.channel[1] + widths[0] * heights[0],
                          dest_gpu.begin() + widths[0] * heights[0] * 1, imagenet_g_char_to_float());
        thrust::transform(thrust::device, image.channel[2], image.channel[2] + widths[0] * heights[0],
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
    if (dest_size.first == -1) {
        return {dest, {1, {*widths, *heights}}};
    }
    double scale;
    NppiRect srcROI = {0, 0, *widths, *heights};
    NppiRect dstROI;
    if ((*widths) <= dest_size.first && (*heights) <= dest_size.second) {
        dstROI = {0, 0, *widths, *heights};
        scale = 1.0f;
    } else {
        if ((*widths) / dest_size.first > (*heights) / dest_size.second) {
            scale = static_cast<double>(dest_size.first) / (*widths);
            dstROI = {0, 0, dest_size.first, static_cast<int>(dest_size.second * scale)};
        } else {
            scale = static_cast<double>(dest_size.second) / (*heights);
            dstROI = {0, 0, static_cast<int>(dest_size.first * scale), dest_size.second};
        }
    }
    NppiSize srcSize = {*widths, *heights};
    NppiSize dstSize = {dest_size.first, dest_size.second};
    Npp32f *resize_gpu_ptr[4];
    Npp32f *resize_gpu_dst_ptr[4];
//    cudaMalloc(reinterpret_cast<void **>(resize_gpu_dst_ptr), dest_size.first * dest_size.second * sizeof(float) * 4);
//    cudaMemset(reinterpret_cast<float *>(resize_gpu_dst_ptr[0]), 1,
//               dest_size.first * dest_size.second * sizeof(float) * 4);
    for (int i = 0; i < 4; ++i) {
        resize_gpu_ptr[i] = thrust::raw_pointer_cast(dest_gpu.data()) + (*widths) * (*heights) * i * sizeof(float);
//        resize_gpu_dst_ptr[i] = resize_gpu_dst_ptr[0] + dest_size.first * dest_size.second * i * sizeof(float);
        cudaMalloc(&resize_gpu_dst_ptr[i], dest_size.first * dest_size.second * sizeof(float));
    }
    CHECK_NPP(nppiResize_32f_P4R(const_cast<const Npp32f **>(resize_gpu_ptr), (*widths) * sizeof(Npp32f), srcSize,
                                 srcROI, resize_gpu_dst_ptr, dest_size.first * sizeof(Npp32f), dstSize,
                                 dstROI, NppiInterpolationMode::NPPI_INTER_LANCZOS))
    vector<float> dest_resized(dest_size.first * dest_size.second * 3);
    auto *dest_malloc = static_cast<float *>(malloc(sizeof(float) * dest_size.first * dest_size.second * 3));
    for (int i = 0; i < 3; ++i) {
        cudaMemcpy(dest_malloc + sizeof(float) * dest_size.first * dest_size.second * i, resize_gpu_dst_ptr[i],
                   dest_size.first * dest_size.second * sizeof(Npp32f),
                   cudaMemcpyKind::cudaMemcpyDeviceToHost);

    }
//    std::copy(dest_malloc, dest_malloc + dest_size.first * dest_size.second * 3, dest_resized.begin());
    for (int i = 0; i < dest_size.first * dest_size.second * 3; ++i) {
        dest_resized[i] = dest_malloc[i];
    }
    free(dest_malloc);
    for (auto &i: resize_gpu_dst_ptr) {
        cudaFree(i);
    }
    auto return_size = dest_size;
    cudaDeviceSynchronize();
    return {dest_resized, {scale, {dest_size.first, dest_size.second}}};

}


using namespace std;
namespace fs = filesystem;
string source_dir = R"(D:\helloproject-ai-data\blog_images)";


int main() {
//    nvjpegStatus_t status;
    int cnt = 0;
    vector<fs::directory_entry> member_names;
    for (auto &person_name: fs::directory_iterator(source_dir)) {
        for (auto &file_name: fs::directory_iterator(person_name)) {
//            cout << file_name.path().string() << endl;
            member_names.push_back(file_name);
        }
    }
    for (const auto &file: member_names) {
        ifstream ifs(file.path(), ios::binary);
//        ifs.seekg(0, ios::end);
//        auto file_length = ifs.tellg();
//        ifs.seekg(0, ios::beg);
//    istream_iterator<unsigned char> begin(ifs), end;
        vector<unsigned char> file_content(
                (istreambuf_iterator<char>(ifs)),
                istreambuf_iterator<char>()
        );

        ifs.close();
        cnt++;
        auto raw_img = decode_image(
                file_content, "none",
                {1080, 1080});
//        cudaFree(raw_img.first);
        cout << file.path() << endl;
        cout << raw_img.second.second.first << endl;
        cout << raw_img.second.second.second << endl;
        if (cnt == 50) {
            exit(0);
        }
    }
    return 0;
}
