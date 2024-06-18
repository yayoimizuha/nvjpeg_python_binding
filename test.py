import time
import cupy_backends.cuda.api.runtime
from simple_decode import decode
from numpy import fromfile, uint8, float32
from cupy.cuda import UnownedMemory, MemoryPointer, set_allocator
from cupy import ndarray, stack, pad, ascontiguousarray
from cupyx.scipy.ndimage import zoom
from torch import Tensor, from_dlpack
from os import listdir, path
from more_itertools import chunked
from onnxruntime import InferenceSession

set_allocator()
IMAGE_SIZE = 640
BATCH_SIZE = 32


def resize_image(image: ndarray) -> ndarray:
    if image.shape[1] <= IMAGE_SIZE and image.shape[2] <= IMAGE_SIZE:
        pass
    else:
        scale = ((IMAGE_SIZE - 1) / max(image.shape[1], image.shape[2]))
        image = zoom(image, (1, scale, scale), mode="constant")
    ret = pad(image, pad_width=[
        (0, 0),
        (0, IMAGE_SIZE - image.shape[1]),
        (0, IMAGE_SIZE - image.shape[2])
    ])
    return ret


root_dir = r"D:\helloproject-ai-data\blog_images"
model_path = r"C:\Users\tomokazu\PycharmProjects\helloproject-ai\test_script\retinaface.onnx"
session = InferenceSession(
    path_or_bytes=model_path,
    providers=[
        'TensorrtExecutionProvider',
        'CUDAExecutionProvider',
        'CPUExecutionProvider'
    ]
)
# from matplotlib import pyplot
files = []
for name in listdir(root_dir):
    for file_name in listdir(path.join(root_dir, name)):
        files.append(path.join(root_dir, name, file_name))
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
        tens: Tensor = from_dlpack(gpu_image)
        print(file, tens.size(), time.time() - start)
        # tens.cpu()
    max_height = max([i.shape[1] for i in stacks])
    max_width = max([i.shape[2] for i in stacks])
    if stacks.__len__() != BATCH_SIZE:
        stacks.extend([ndarray([3, IMAGE_SIZE, IMAGE_SIZE])] * (BATCH_SIZE - stacks.__len__()))
    stacked_images = stack([resize_image(gpu_image) for gpu_image in stacks])
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
    [cupy_backends.cuda.api.runtime.free(ptr) for ptr in ptrs]
# host_im = gpu_image.get()
# pyplot.imshow(host_im.transpose((1, 2, 0)))
# pyplot.show()
