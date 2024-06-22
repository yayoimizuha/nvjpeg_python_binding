import time
import cupy_backends.cuda.api.runtime
from numba import njit, prange, types, int64
from simple_decode import decode
from numpy import fromfile, uint8, float32
import numpy as np
from cupy.cuda import UnownedMemory, MemoryPointer, set_allocator, stream
from cupy import ndarray, stack, pad, ascontiguousarray
from cupyx.scipy.ndimage import zoom
from torch import Tensor, from_dlpack
from os import listdir, path
from more_itertools import chunked
from onnxruntime import InferenceSession
import math

set_allocator()
IMAGE_SIZE = 640
BATCH_SIZE = 16


def resize_image(image: ndarray, scales: list[float]) -> ndarray:
    if image.shape[1] <= IMAGE_SIZE and image.shape[2] <= IMAGE_SIZE:
        scale = 1.0
        pass
    else:
        scale = ((IMAGE_SIZE - 1) / max(image.shape[1], image.shape[2]))
        image = zoom(image, (1, scale, scale), mode="constant")
    ret = pad(image, pad_width=[
        (0, 0),
        (0, IMAGE_SIZE - image.shape[1]),
        (0, IMAGE_SIZE - image.shape[2])
    ])
    scales.append(scale)
    return ret


# @njit()
def prior_box(min_sizes: list[list[int]], steps: list[int], clip: bool, image_sizes: list[int]) -> np.ndarray:
    feature_maps = [[math.ceil(image_sizes[0] / step), math.ceil(image_sizes[1] / step)] for step in steps]
    anchors = []
    for k, f in enumerate(feature_maps):
        min_sizes_k = min_sizes[k]
        for i in range(f[0]):
            for j in range(f[1]):
                for min_size in min_sizes_k:
                    s_kx = min_size / image_sizes[1]
                    s_ky = min_size / image_sizes[0]
                    dense_cx = [x * steps[k] / image_sizes[1] for x in [j + 0.5]]
                    dense_cy = [y * steps[k] / image_sizes[0] for y in [i + 0.5]]
                    for cy in dense_cy:
                        for cx in dense_cx:
                            anchors.append([cx, cy, s_kx, s_ky])
                    # for cy, cx in np.nditer([np.array(dense_cy), np.array(dense_cx).reshape((-1, 1))]):

    output = np.array(anchors)
    if clip:
        output.clip(.0, 1.0)
    return output


# @njit
def loc_decode(loc: np.ndarray, priors: np.ndarray, variances: list[float]):
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


# @njit
def decode_landm(pre: np.ndarray, priors: np.ndarray, variances: list[float]):
    return np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                           priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                           priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                           priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                           priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                           ), axis=1)


# @njit
def py_cpu_nms(dets: np.ndarray, thresh: float):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


# @njit
def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# @njit
def post_process(landms: np.ndarray, conf: np.ndarray, loc: np.ndarray, image_sizes: list[int], resize_scale: float,
                 confidence_threshold: float, top_k: int, nms_threshold: float, keep_top_k: int) -> np.ndarray:
    priors = prior_box(min_sizes=[[16, 32], [64, 128], [256, 512]], steps=[8, 16, 32], clip=False,
                       image_sizes=image_sizes)
    boxes = loc_decode(loc, priors, [0.1, 0.2])
    boxes_scale = np.array([image_sizes[1], image_sizes[0]] * 2)
    boxes = boxes * boxes_scale / resize_scale

    conf = softmax(conf)
    scores = conf[:, 1]
    landms = decode_landm(landms, priors, [0.1, 0.2])
    landms_scale = np.array([image_sizes[1], image_sizes[0]] * 5)
    landms = landms * landms_scale / resize_scale

    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]

    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    return dets


# @njit(parallel=True)
def post_loop(landms: np.ndarray, conf: np.ndarray, loc: np.ndarray, resize_scales: list[float]):
    res = []
    for i in range(BATCH_SIZE):
        rep = post_process(landms=landms[i, :, :], conf=conf[i, :, :], loc=loc[i, :, :],
                           image_sizes=[IMAGE_SIZE, IMAGE_SIZE],
                           resize_scale=resize_scales[i], confidence_threshold=0.4, top_k=5000, nms_threshold=0.4,
                           keep_top_k=750)
        res.append(rep)
    return res


root_dir = r"D:\helloproject-ai-data\blog_images"
model_path = r"C:\Users\tomokazu\build\retinaface\retinaface_only_nn.onnx"
session = InferenceSession(
    path_or_bytes=model_path,
    providers=[
        ('TensorrtExecutionProvider', {
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': 'trt_cache',
            'trt_fp16_enable': True,
            'trt_profile_min_shapes': 'input:1x3x640x640',
            'trt_profile_max_shapes': 'input:32x3x640x640',
            'trt_profile_opt_shapes': 'input:32x3x640x640',
        }),
        'CUDAExecutionProvider',
        'CPUExecutionProvider'
    ]
)
# from matplotlib import pyplot
files = []
for name in listdir(root_dir):
    if name != "後藤花":
        continue
        pass
    for file_name in listdir(path.join(root_dir, name)):
        files.append(path.join(root_dir, name, file_name))
files = files[:32]

for file_chunk in chunked(files, BATCH_SIZE):
    stacks = []
    ptrs = []
    for file in file_chunk:
        start = time.time()
        _input = fromfile(file=file, dtype=uint8)
        if _input.shape[0] == 0:
            continue
        ptr, (_, (width, height)) = decode(_input)
        ptrs.append(ptr)
        # print(ptr, width, height)
        unownedmemory = UnownedMemory(ptr, height * width * 3 * uint8().itemsize, None)
        gpu_arr = ndarray((height * width * 3,), dtype=float32, memptr=MemoryPointer(unownedmemory, 0))
        gpu_image: ndarray = gpu_arr.reshape((3, height, width))
        stacks.append(gpu_image)
        # tens: Tensor = from_dlpack(gpu_image)
        print(file, gpu_image.shape, time.time() - start, sep='\t')
        # tens.cpu()
    max_height = max([i.shape[1] for i in stacks])
    max_width = max([i.shape[2] for i in stacks])
    if stacks.__len__() != BATCH_SIZE:
        stacks.extend([ndarray([3, IMAGE_SIZE, IMAGE_SIZE])] * (BATCH_SIZE - stacks.__len__()))
    st = stream.Stream()
    resize_scales = []
    with st:
        stacked_images = stack([resize_image(gpu_image, resize_scales) for gpu_image in stacks])
    st.synchronize()
    print(stacked_images.shape)
    contiguous_stacked = ascontiguousarray(stacked_images)
    io_binding = session.io_binding()
    io_binding.bind_input(
        name="input",
        device_type='cuda',
        device_id=stacked_images.device,
        element_type=float32,
        shape=tuple(stacked_images.shape),
        buffer_ptr=contiguous_stacked.data.ptr
    )
    io_binding.bind_output("landmark")
    io_binding.bind_output("confidence")
    io_binding.bind_output("bbox")
    session.run_with_iobinding(iobinding=io_binding)
    landms, conf, loc = io_binding.copy_outputs_to_cpu()
    detected = post_loop(landms, conf, loc, resize_scales)
    for i in detected:
        for j in i.tolist():
            print(i[0])

    [cupy_backends.cuda.api.runtime.free(ptr) for ptr in ptrs]
# host_im = gpu_image.get()
# pyplot.imshow(host_im.transpose((1, 2, 0)))
# pyplot.show()
