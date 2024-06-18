#include <iostream>
#include <nvjpeg.h>
#include <string>
#include <filesystem>
#include <algorithm>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <pybind11/stl.h>

using namespace std;
namespace fs = filesystem;
string source_dir = R"(D:\helloproject-ai-data\blog_images)";
#define CHECK_CUDA(call) {cudaError_t _e = (call); if (_e != cudaSuccess){throw runtime_error("CUDA Runtime error at "s+__FILE__+":"s+to_string(__LINE__)+" as #"s+to_string(static_cast<int>(_e)));}}
#define CHECK_NVJPEG(call) {nvjpegStatus_t _e = (call);if (_e != NVJPEG_STATUS_SUCCESS){throw runtime_error("NVJPEG error at "s+__FILE__+":"s+to_string(__LINE__)+" as #"s+to_string(static_cast<int>(_e)));}}
#define CHECK_NPP(call) {NppStatus _e = (call);if (_e != NppStatus::NPP_SUCCESS){throw runtime_error("NPP error at "s+__FILE__+":"s+to_string(__LINE__)+" as #"s+to_string(static_cast<int>(_e)));}};

bool nvInit = false;
cudaStream_t stream = nullptr;
nvjpegJpegState_t state = nullptr;
nvjpegHandle_t handle = nullptr;

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

static_assert(sizeof(size_t) >= sizeof(float *), "cannot cast pointer.");

pair<size_t, pair<double, pair<int, int>>> decode_image(const vector<unsigned char> &file_data) {
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

//    thrust::device_vector<float> dest_gpu(3 * widths[0] * heights[0]);
    float *dest_gpu;
    CHECK_CUDA(cudaMalloc(&dest_gpu, 3 * widths[0] * heights[0] * sizeof(float)))
    for (int i = 0; i < 3; i++) {
        thrust::transform(thrust::device, image.channel[i], image.channel[i] + widths[0] * heights[0],
                          dest_gpu + widths[0] * heights[0] * i, none_char_to_float());
    }

    for (auto &i: image.channel) {
        CHECK_CUDA(cudaFree(i))
    }
    auto dest_ptr = reinterpret_cast<size_t>(dest_gpu);
    auto return_val = make_pair(dest_ptr, make_pair(1.0, make_pair(*widths, *heights)));

    return return_val;

}


PYBIND11_MODULE(simple_decode, m) {
    m.def("decode", &decode_image, "pass image binary and return decoded gpu pointer.");
}
//
//int main() {
////    nvjpegStatus_t status;
//    int cnt = 0;
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
//        cnt++;
//        auto raw_img = decode_image(
//                file_content, "none",
//                {2000, 2000});
//        cudaFree(raw_img.first);
//        cout << file.path() << endl;
//        cout << raw_img.second.second.first << endl;
//        cout << raw_img.second.second.second << endl;
//        if (cnt == 50) {
////            exit(0);
//        }
//    }
//    return 0;
//}
