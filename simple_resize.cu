#include <iostream>
#include <nvjpeg.h>
#include <string>
#include <filesystem>
#include <algorithm>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <fstream>

using namespace std;
namespace fs = filesystem;
#define CHECK_CUDA(call) {cudaError_t _e = (call); if (_e != cudaSuccess){throw runtime_error("CUDA Runtime error at "s+__FILE__+":"s+to_string(__LINE__)+" as #"s+to_string(static_cast<int>(_e)));}}
#define CHECK_NVJPEG(call) {nvjpegStatus_t _e = (call);if (_e != NVJPEG_STATUS_SUCCESS){throw runtime_error("NVJPEG error at "s+__FILE__+":"s+to_string(__LINE__)+" as #"s+to_string(static_cast<int>(_e)));}}
#define CHECK_NPP(call) {NppStatus _e = (call);if (_e != NppStatus::NPP_SUCCESS){throw runtime_error("NPP error at "s+__FILE__+":"s+to_string(__LINE__)+" as #"s+to_string(static_cast<int>(_e)));}};


pair<vector<float>, pair<double, pair<int, int>>>
decode_image(const vector<unsigned char> &file_data, const string &normalize, pair<int, int> dest_size) {

}

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
                {2000, 2000});
//        cudaFree(raw_img.first);
        cout << file.path() << endl;
        cout << raw_img.second.second.first << endl;
        cout << raw_img.second.second.second << endl;
        if (cnt == 50) {
//            exit(0);
        }
    }
    return 0;
}
