import json
import math
import platform
import warnings
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.cuda import amp

from utils.datasets import exif_transpose, letterbox
from utils.general import (LOGGER, check_requirements, check_suffix, check_version, colorstr, increment_path,
                           make_divisible, non_max_suppression, scale_coords, xywh2xyxy, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, time_sync


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
    def forward(self, x):
        return torch.cat(x, self.d)



class DetectMultiBackend(nn.Module):
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False):   
        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs = self.model_type(w)  # get backend
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        w = attempt_download(w)  # download if not local
        fp16 &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16
        if data:  # data.yaml path (optional)
            with open(data, errors='ignore') as f:
                names = yaml.safe_load(f)['names']  # class names

        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, map_location=device)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif jit:  # TorchScript
            LOGGER.info(f'Loading {w} for TorchScript inference...')
            extra_files = {'config.txt': ''}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files)
            model.half() if fp16 else model.float()
            if extra_files['config.txt']:
                d = json.loads(extra_files['config.txt'])  # extra_files dict
                stride, names = int(d['stride']), d['names']
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f'Loading {w} for ONNX OpenCV DNN inference...')
            check_requirements(('opencv-python>=4.5.4',))
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            cuda = torch.cuda.is_available()
            check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(w, providers=providers)
        elif xml:  # OpenVINO
            LOGGER.info(f'Loading {w} for OpenVINO inference...')
            check_requirements(('openvino-dev',))  # requires openvino-dev: https://pypi.org/project/openvino-dev/
            import openvino.inference_engine as ie
            core = ie.IECore()
            if not Path(w).is_file():  # if not *.xml
                w = next(Path(w).glob('*.xml'))  # get *.xml file from *_openvino_model dir
            network = core.read_network(model=w, weights=Path(w).with_suffix('.bin'))  # *.xml, *.bin paths
            executable_network = core.load_network(network, device_name='CPU', num_requests=1)
        elif engine:  # TensorRT
            LOGGER.info(f'Loading {w} for TensorRT inference...')
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
            check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            bindings = OrderedDict()
            fp16 = False  # default updated below
            for index in range(model.num_bindings):
                name = model.get_binding_name(index)
                dtype = trt.nptype(model.get_binding_dtype(index))
                shape = tuple(model.get_binding_shape(index))
                data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
                bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
                if model.binding_is_input(index) and dtype == np.float16:
                    fp16 = True
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            context = model.create_execution_context()
            batch_size = bindings['images'].shape[0]
        elif coreml:  # CoreML
            LOGGER.info(f'Loading {w} for CoreML inference...')
            import coremltools as ct
            model = ct.models.MLModel(w)
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            if saved_model:  # SavedModel
                LOGGER.info(f'Loading {w} for TensorFlow SavedModel inference...')
                import tensorflow as tf
                keras = False  # assume TF1 saved_model
                model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
            elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
                LOGGER.info(f'Loading {w} for TensorFlow GraphDef inference...')
                import tensorflow as tf

                def wrap_frozen_graph(gd, inputs, outputs):
                    x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                    ge = x.graph.as_graph_element
                    return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

                gd = tf.Graph().as_graph_def()  # graph_def
                gd.ParseFromString(open(w, 'rb').read())
                frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs="Identity:0")
            elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
                try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                    from tflite_runtime.interpreter import Interpreter, load_delegate
                except ImportError:
                    import tensorflow as tf
                    Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,
                if edgetpu:  # Edge TPU https://coral.ai/software/#edgetpu-runtime
                    LOGGER.info(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
                    delegate = {'Linux': 'libedgetpu.so.1',
                                'Darwin': 'libedgetpu.1.dylib',
                                'Windows': 'edgetpu.dll'}[platform.system()]
                    interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
                else:  # Lite
                    LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
                    interpreter = Interpreter(model_path=w)  # load TFLite model
                interpreter.allocate_tensors()  # allocate
                input_details = interpreter.get_input_details()  # inputs
                output_details = interpreter.get_output_details()  # outputs
            elif tfjs:
                raise Exception('ERROR: YOLOv5 TF.js inference is not supported')
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, val=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.pt or self.jit:  # PyTorch
            y = self.model(im) if self.jit else self.model(im, augment=augment, visualize=visualize)
            return y if val else y[0]
        elif self.dnn:  # ONNX OpenCV DNN
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})[0]
        elif self.xml:  # OpenVINO
            im = im.cpu().numpy()  # FP32
            desc = self.ie.TensorDesc(precision='FP32', dims=im.shape, layout='NCHW')  # Tensor Description
            request = self.executable_network.requests[0]  # inference request
            request.set_blob(blob_name='images', blob=self.ie.Blob(desc, im))  # name=next(iter(request.input_blobs))
            request.infer()
            y = request.output_blobs['output'].buffer  # name=next(iter(request.output_blobs))
        elif self.engine:  # TensorRT
            assert im.shape == self.bindings['images'].shape, (im.shape, self.bindings['images'].shape)
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = self.bindings['output'].data
        elif self.coreml:  # CoreML
            im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            im = Image.fromarray((im[0] * 255).astype('uint8'))
            # im = im.resize((192, 320), Image.ANTIALIAS)
            y = self.model.predict({'image': im})  # coordinates are xywh normalized
            if 'confidence' in y:
                box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
                conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float)
                y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            else:
                k = 'var_' + str(sorted(int(k.replace('var_', '')) for k in y)[-1])  # output key
                y = y[k]  # output
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            if self.saved_model:  # SavedModel
                y = (self.model(im, training=False) if self.keras else self.model(im)).numpy()
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im)).numpy()
            else:  # Lite or Edge TPU
                input, output = self.input_details[0], self.output_details[0]
                int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input['index'], im)
                self.interpreter.invoke()
                y = self.interpreter.get_tensor(output['index'])
                if int8:
                    scale, zero_point = output['quantization']
                    y = (y.astype(np.float32) - zero_point) * scale  # re-scale
            y[..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, device=self.device)
        return (y, []) if val else y

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        if any((self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb)):  # warmup types
            if self.device.type != 'cpu':  # only warmup GPU models
                im = torch.zeros(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
                for _ in range(2 if self.jit else 1):  #
                    self.forward(im)  # warmup

    @staticmethod
    def model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        from export import export_formats
        suffixes = list(export_formats().Suffix) + ['.xml']  # export suffixes
        check_suffix(p, suffixes)  # checks
        p = Path(p).name  # eliminate trailing separators
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, xml2 = (s in p for s in suffixes)
        xml |= xml2  # *_openvino_model or *.xml
        tflite &= not edgetpu  # *.tflite
        return pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs


class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    amp = False  # Automatic Mixed Precision (AMP) inference

    def __init__(self, model):
        super().__init__()
        LOGGER.info('Adding AutoShape... ')
        copy_attr(self, model, include=('yaml', 'nc', 'hyp', 'names', 'stride', 'abc'), exclude=())  # copy attributes
        self.dmb = isinstance(model, DetectMultiBackend)  # DetectMultiBackend() instance
        self.pt = not self.dmb or model.pt  # PyTorch model
        self.model = model.eval()

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_sync()]
        p = next(self.model.parameters()) if self.pt else torch.zeros(1)  # for device and type
        autocast = self.amp and (p.device.type != 'cpu')  # Automatic Mixed Precision (AMP) inference
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=autocast):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, (str, Path)):  # filename or uri
                im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                im = np.asarray(exif_transpose(im))
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, self.stride) if self.pt else size for x in np.array(shape1).max(0)]  # inf shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32
        t.append(time_sync())

        with amp.autocast(enabled=autocast):
            # Inference
            y = self.model(x, augment, profile)  # forward
            t.append(time_sync())

            # Post-process
            y = non_max_suppression(y if self.dmb else y[0], self.conf, iou_thres=self.iou, classes=self.classes,
                                    agnostic=self.agnostic, multi_label=self.multi_label, max_det=self.max_det)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_sync())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, imgs, pred, files, times=(0, 0, 0, 0), names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.times = times  # profiling times
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        crops = []
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            s = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({'box': box, 'conf': conf, 'cls': cls, 'label': label,
                                          'im': save_one_box(box, im, file=file, save=save)})
                        else:  # all others
                            annotator.box_label(box, label, color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                LOGGER.info(s.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.imgs[i] = np.asarray(im)
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    def print(self):
        self.display(pprint=True)  # print results
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' %
                    self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True) if save else None
        return self.display(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        r = range(self.n)  # iterable
        x = [Detections([self.imgs[i]], [self.pred[i]], [self.files[i]], self.times, self.names, self.s) for i in r]
        # for d in x:
        #    for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
        #        setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)
class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
         return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


##########################################################################################################################
from functools import partial
from typing import Iterable, Tuple, Union

import torch
import torch.nn.functional as f
from torch import Tensor, nn
from torch.fft import irfftn, rfftn


def complex_matmul(a: Tensor, b: Tensor, groups: int = 1) -> Tensor:
    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors, over only the first channel
    # dimensions. Dimensions 3 and higher will have the same shape after multiplication.
    # We also allow for "grouped" multiplications, where multiple sections of channels
    # are multiplied independently of one another (required for group convolutions).
    scalar_matmul = partial(torch.einsum, "agc..., gbc... -> agb...")
    a = a.view(a.size(0), groups, -1, *a.shape[2:])
    b = b.view(groups, -1, *b.shape[1:])

    # Compute the real and imaginary parts independently, then manually insert them
    # into the output Tensor.  This is fairly hacky but necessary for PyTorch 1.7.0,
    # because Autograd is not enabled for complex matrix operations yet.  Not exactly
    # idiomatic PyTorch code, but it should work for all future versions (>= 1.7.0).
    real = scalar_matmul(a.real, b.real) - scalar_matmul(a.imag, b.imag)
    imag = scalar_matmul(a.imag, b.real) + scalar_matmul(a.real, b.imag)
    c = torch.zeros(real.shape, dtype=torch.complex64, device=a.device)
    c.real, c.imag = real, imag

    return c.view(c.size(0), -1, *c.shape[3:])


def to_ntuple(val: Union[int, Iterable[int]], n: int) -> Tuple[int, ...]:
    if isinstance(val, Iterable):
        out = tuple(val)
        if len(out) == n:
            return out
        else:
            raise ValueError(f"Cannot cast tuple of length {len(out)} to length {n}.")
    else:
        return n * (val,)


def fft_conv(
    signal: Tensor,
    kernel: Tensor,
    bias: Tensor = None,
    padding: Union[int, Iterable[int], str] = 0,
    stride: Union[int, Iterable[int]] = 1,
    groups: int = 1,
    padding_mode: str = "constant",
) -> Tensor:
    # Cast stride to tuple.
    stride_ = to_ntuple(stride, n=signal.ndim - 2)

    if padding != "same":
        padding_ = to_ntuple(padding, n=signal.ndim - 2)
        signal_padding = [p for p in padding_[::-1] for _ in range(2)]
    else:
        # signal_padding = [
        #     (0, 0) if k <= s else ((k - s) // 2, k - (k - s) // 2)
        #     for s, k, in zip(signal.shape[2:], kernel.shape[2:])
        # ]
        # signal_padding = [p for pd in signal_padding[::-1] for p in pd]
        padding_ = [k // 2 for k in kernel.shape[2:]]

    signal_padding = [p for p in padding_[::-1] for _ in range(2)]
    # Pad the input signal & kernel tensors
    signal = f.pad(signal, signal_padding, mode=padding_mode)

    # Because PyTorch computes a *one-sided* FFT, we need the final dimension to
    # have *even* length.  Just pad with one more zero if the final dimension is odd.
    signal_size = signal.size()  # original signal size without padding to even
    if signal.size(-1) % 2 != 0:
        signal = f.pad(signal, [0, 1])

    kernel_padding = [
        pad
        for i in reversed(range(2, signal.ndim))
        for pad in [0, signal.size(i) - kernel.size(i)]
    ]

    padded_kernel = f.pad(kernel, kernel_padding)
    assert (
        padded_kernel.shape[1:] == signal.shape[1:]
    ), f"padded kernel shape {padded_kernel.shape} not equal to signal shape {signal.shape}"

    # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    # signal = signal.reshape(signal.size(0), groups, -1, *signal.shape[2:])
    signal_fr = rfftn(signal.float(), dim=tuple(range(2, signal.ndim)))
    kernel_fr = rfftn(padded_kernel.float(), dim=tuple(range(2, signal.ndim)))

    kernel_fr.imag *= -1
    output_fr = complex_matmul(signal_fr, kernel_fr, groups=groups)
    output = irfftn(output_fr, dim=tuple(range(2, signal.ndim)))

    # Remove extra padded values
    crop_slices = [slice(None), slice(None)] + [
        slice(
            0,
            (signal_size[i] - kernel.size(i) + (kernel.size(i) % 2)),
            # if padding != "same"
            # else None,
            stride_[i - 2],
        )
        for i in range(2, signal.ndim)
    ]
    output = output[crop_slices].contiguous()

    # Optionally, add a bias term before returning.
    if bias is not None:
        bias_shape = tuple([1, -1] + (signal.ndim - 2) * [1])
        output += bias.view(bias_shape)

    return output
class TransformerBlock1(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = _FFTConv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)
class Bottleneck1(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = _FFTConv(c1, c_, 1, 1)
        self.cv2 = _FFTConv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))




class _FFTConv(nn.Module):
    """Base class for PyTorch FFT convolution layers."""

    def __init__(self,in_channels: int,out_channels: int,kernel_size: Union[int, Iterable[int]],
        padding: Union[int, Iterable[int]] = 0,stride: Union[int, Iterable[int]] = 1,
        groups: int = 1,bias: bool = True,ndim: int = 2,):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.groups = groups
        self.use_bias = bias

        if in_channels % groups != 0:
            raise ValueError(
                "'in_channels' must be divisible by 'groups'."
                f"Found: in_channels={in_channels}, groups={groups}."
            )
        if out_channels % groups != 0:
            raise ValueError(
                "'out_channels' must be divisible by 'groups'."
                f"Found: out_channels={out_channels}, groups={groups}."
            )

        kernel_size = to_ntuple(kernel_size, ndim)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        self.bias = (nn.Parameter(torch.randn(out_channels,) )
            if bias
            else None
        )

    def forward(self, signal):
        return fft_conv( signal,self.weight,bias=self.bias, padding=self.padding, stride=self.stride, groups=self.groups, )
1
class FC1(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, cin):
        super().__init__()
        self.act = nn.Softmax(dim=-1)
        self.c = _FFTConv(in_channels=cin, out_channels=cin, kernel_size=1, stride=1, bias=True)
        self.lc = _FFTConv(in_channels=cin, out_channels=cin, kernel_size=3, stride=1, padding=1,bias=True)
        self.sc = _FFTConv(in_channels=cin, out_channels=cin, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        q, k, v = self.c(x), self.c(x), self.c(x)
        q_r, k_r, v_r = q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)
        _, _, h, w = x.shape

        x_row = self.act(torch.matmul(q, k_r) * w ** -0.5)
        x_col = self.act(torch.matmul(q_r, k) * h ** -0.5)
        x_row = self.lc(self.sc(torch.matmul(x_row, v)))
        x_col = self.lc(self.sc(torch.matmul(x_col, v_r)))

        return x + x_row + x_col.transpose(2, 3)


class C31(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1,  e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c1 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = _FFTConv(c1, c_, 3, 1)
        self.cv3 = Conv(2 * c_, c1, 1)  # act=FReLU(c2)

    def forward(self, x):
        return self.cv3(torch.cat(self.cv1(x), self.cv2(x), dim=1))

class DyReLU(nn.Module):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLU, self).__init__()
        self.channels = channels
        self.k = k
        self.conv_type = conv_type
        assert self.conv_type in ['1d', '2d']

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2*k)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.]*k + [0.5]*k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.]*(2*k - 1)).float())

    def get_relu_coefs(self, x):
        theta = torch.mean(x, axis=-1)
        if self.conv_type == '2d':
            theta = torch.mean(theta, axis=-1)
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x):
        raise NotImplementedError

class DyReLUA(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUA, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, 2*self.k) * self.lambdas + self.init_v
        # BxCxL -> LxCxBx1
        x_perm = x.transpose(0, -1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :self.k] + relu_coefs[:, self.k:]
        # LxCxBx2 -> BxCxL
        result = torch.max(output, dim=-1)[0].transpose(0, -1)

        return result
class resBottleneck1(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, groups=1, base_width=64, dilation=1, norm_layer=None, downsample=False,use=False):
        super(resBottleneck1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)

        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if downsample:
            self.downsample = nn.Sequential(conv1x1(inplanes, planes * self.expansion, stride),nn.BatchNorm2d(planes * self.expansion),)
        else:
            self.downsample=None
        if use:
            #self.use = _FFTConv(width,width,3,1,1) C31
            self.use = C31(width)

        else:
            self.use=None
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        if self.use is not None:
            out = self.use(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CFFC(nn.Module):
    def __init__(self, c1, c2, n=1, s=1, g=1, w=64, dilation=1,norm_layer=None,downsample=True):
        super(CFFC,self).__init__()
        blocks=[resBottleneck1(c1, c2, stride=s, groups=g, base_width=w,  dilation=1,norm_layer=None,downsample=downsample)]
        for _ in range(n-1):
            blocks.append(resBottleneck1(inplanes=c2*resBottleneck1.expansion, planes=c2, stride=1, groups=g, base_width=w))
        self.layers = nn.Sequential(*blocks)
    def forward(self, x):
        return self.layers(x)

############################################################################################################################################
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard, block_diag



class TransformBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        bn=True,
        dropout=True,
        kernel_size=3,
        padding=1,
        stride=1,
        alpha_root=None,
        add_noise=False,
        lmbda=None,
        diag=False,
        bias=False,
        use_res=True,
    ):
        super().__init__()
        self.bn = bn  # flag for batchnorm (True/False)
        self.drop = dropout  # flag for dropout (True/False)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.bias = bias  # flag for bias
        self.diag = diag  # select only diagonal filters
        self.use_res = use_res  # use residual connection or not
        self.alpha_root = alpha_root  # use rooting, i.e. (input.pow(alpha))
        # Filters based on transform
        self.filter_bank = None
        # Linear Combination conv
        self.conv = None
        # Shortcut is the downsampling layer. It preserves the input info
        self.shortcut = nn.Identity()
        # if use_res is False,
        # it is just identity mapping
        if (input_channels != output_channels or stride != 1) and self.use_res:
            ks = 2 if self.kernel_size % 2 == 0 else 1
            p = 1 if ks == 2 else 0
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    padding=p,
                    kernel_size=ks,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
            )

        if add_noise:
            self.noise = nn.Parameter(
                nn.init.normal_(
                    torch.randn(1, self.filter_bank.size(0)), std=1e-4
                )
            )
        else:
            self.noise = None
        if lmbda is not None:
            # limits the number of kernels
            self.lmbda = min(lmbda, self.kernel_size ** 2)
        else:
            self.lmbda = lmbda

    @classmethod
    def get_idx(self, ker: int, lmbda: int):
        """Return indices for partial filter usage, see Ulicny et al. 2018."""
        out = []
        for i in range(ker):
            for j in range(ker):
                if i + j < lmbda:
                    out.append(ker * i + j)
        return tuple(out)

    @classmethod
    def get_idx_diag(self, ker):
        """Return only diagonal indices for partial filter usage."""
        out = []
        for i in range(ker):
            for j in range(ker):
                if i == j:
                    out.append(i + j)
        return tuple(out)

    @classmethod
    def draw_filters(self, fb_=None, figsize=(12, 4)):
        if not fb_:
            fb_ = self.filter_bank
        fig, ax = plt.subplots(len(fb_), 1, figsize=figsize)
        for i in range(len(fb_)):
            ax[i].imshow(fb_[i, 0, :, :])
            ax[i].axis("off")
            ax[i].grid(False)

    def filter_from_matrix(self, i, j, size):
        raise NotImplementedError

    def get_filter_bank(
        self, kernel_size, input_channels=3, lmbda=None, diag=False, **kwargs
    ):
        """Build filter bank from a matrix.
        Parameters:
             :kernel_size: integer, determines sizes of filters.
             input
        """
        filter_bank = torch.zeros(
            (kernel_size, kernel_size, kernel_size, kernel_size)
        )
        for i in range(kernel_size):
            for j in range(kernel_size):
                filter_bank[i, j, :, :] = self.filter_from_matrix(
                    i=i, j=j, size=kernel_size
                )
        if lmbda:
            ids = self.get_idx(kernel_size, lmbda)
            return torch.stack(
                tuple(
                    [
                        (filter_bank.view(-1, 1, kernel_size, kernel_size))[
                            ids, ...
                        ]
                    ]
                    * input_channels
                ),
                dim=0,
            ).view((-1, 1, kernel_size, kernel_size))
        if diag:
            ids = self.get_idx_diag(kernel_size)
            return torch.stack(
                tuple(
                    [
                        filter_bank.view(-1, 1, kernel_size, kernel_size)[
                            ids, ...
                        ]
                    ]
                    * input_channels
                ),
                dim=0,
            ).view((-1, 1, kernel_size, kernel_size))
        return torch.stack(
            tuple(
                [filter_bank.view(-1, 1, kernel_size, kernel_size)]
                * input_channels
            ),
            dim=0,
        ).view((-1, 1, kernel_size, kernel_size))

    @classmethod
    def alpha_rooting(self, x, alpha=1.0):
        if alpha is not None:
            return x.sign() * torch.abs(x).pow(alpha)
        else:
            return x


class HarmonicBlock(TransformBlock):
    def __init__(
        self,
        *args,
        type=2,
        droprate=0.5,
        **kwargs,
    ):
        super(HarmonicBlock, self).__init__(*args, **kwargs)
        self.type = type
        self.filter_bank = self.get_filter_bank(
            kernel_size=self.kernel_size,  # kernel size
            input_channels=self.input_channels,
            t=self.type,  # type of DCT
            lmbda=self.lmbda,
            diag=self.diag,
        ).float()
        self.conv = nn.Conv2d(
            in_channels=self.filter_bank.shape[0],
            out_channels=self.output_channels,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=self.bias,
        )
        if self.bn:
            self.bnorm = nn.BatchNorm2d(self.filter_bank.shape[0])
        if self.drop:
            self.dropout = nn.Dropout(droprate)

    @staticmethod
    def dct_matrix(t=2, N=32):
        if t == 1:
            # N- the size of the input
            # n is the column dummy index, k is the row dummy index
            res = np.zeros((N, N))
            res[:, 0] = 0.5
            for p in range(N):
                res[p, -1] = (-1) ** p
            for n in range(1, N - 1):
                for k in range(N):
                    res[k, n] = np.cos(np.pi / (N - 1) * n * k)
            return res
        if t == 2:
            res = np.zeros((N, N))
            for k in range(N):
                for n in range(N):
                    res[k, n] = np.cos(np.pi / (N) * (n + 0.5) * k)
            return res
        if t == 3:
            res = np.zeros((N, N))
            res[:, 0] = 0.5
            for n in range(1, N):
                for k in range(N):
                    res[k, n] = np.cos(np.pi / (N) * n * (k + 0.5))
            return res
        if t == 4:
            res = np.zeros((N, N))
            for k in range(N):
                for n in range(N):
                    res[k, n] = np.cos(np.pi / (N) * (n + 0.5) * (k + 0.5))
            return res

    def filter_from_matrix(self, i, j, size):
        mat = self.dct_matrix(t=self.type, N=size)
        fltr = (
            mat[i, : self.kernel_size]
            .reshape((-1, 1))
            .dot(mat[j, : self.kernel_size].reshape(1, -1))
        )
        return torch.as_tensor(fltr)

    def forward(self, x):
        in_ = x
        x = F.conv2d(
            x.float(),
            weight=self.filter_bank,
            padding=self.padding,
            stride=self.stride,
            groups=self.input_channels,
        )  # int(self.K/
        x = self.alpha_rooting(x, alpha=self.alpha_root)
        if self.noise is not None:
            x.add_(self.noise.unsqueeze(-1).unsqueeze(-1))
        if self.bn:
            x = F.relu(self.bnorm(x))
        else:
            x = F.relu(x)
        if self.drop:
            x = self.dropout(x)
        if self.use_res:
            x = self.conv(x) + self.shortcut(in_)
        else:
            x = self.conv(x)
        x = F.relu(x)
        return x


class SlantBlock(TransformBlock):
    def __init__(self, *args, droprate=0.5, **kwargs):
        super(SlantBlock, self).__init__(*args, **kwargs)
        self.filter_bank = self.get_filter_bank(
            kernel_size=self.kernel_size,  # kernel size
            input_channels=self.input_channels,
            lmbda=self.lmbda,
            diag=self.diag,
        ).float()
        self.conv = nn.Conv2d(
            in_channels=self.filter_bank.shape[0],
            out_channels=self.output_channels,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=self.bias,
        )
        if self.bn:
            self.bnorm = nn.BatchNorm2d(self.filter_bank.shape[0])
        if self.drop:
            self.dropout = nn.Dropout(droprate)

    def filter_from_matrix(self, i, j, size):
        def slant(n):
            N = (2 ** (size - 1)).bit_length()
            S_1 = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])

            an = np.sqrt((2 * N ** 2) / (4 * N ** 2 - 1))  # a2
            bn = np.sqrt((N ** 2 - 1) / (4 * N ** 2 - 1))  # b2

            S_2 = (
                1
                / np.sqrt(2)
                * np.array(
                    [
                        [1, 0, 1, 0],
                        [an, bn, -an, bn],
                        [0, 1, 0, -1],
                        [-bn, an, bn, an],
                    ]
                )
            )

            S_2 = np.matmul(S_2, block_diag(S_1, S_1))

            if N == 2:
                return S_1
            elif N == 4:
                return S_2
            else:
                S_prev = S_2
                i = 3
                while i >= 3 and i <= n:
                    N = 2 ** i
                    an = np.sqrt((3 * N ** 2) / (4 * N ** 2 - 1))  # a2
                    bn = np.sqrt((N ** 2 - 1) / (4 * N ** 2 - 1))
                    An1 = np.array([[1, 0], [an, bn]])
                    An2 = np.array([[1, 0], [-an, bn]])
                    Bn1 = np.array([[0, 1], [-bn, an]])
                    Bn2 = np.array([[0, -1], [bn, an]])
                    S_N = np.block(
                        [
                            [
                                An1,
                                np.zeros((2, N // 2 - 2)),
                                An2,
                                np.zeros((2, N // 2 - 2)),
                            ],
                            [
                                np.zeros((N // 2 - 2, 2)),
                                np.eye(N // 2 - 2),
                                np.zeros((N // 2 - 2, 2)),
                                np.eye(N // 2 - 2),
                            ],
                            [
                                Bn1,
                                np.zeros((2, N // 2 - 2)),
                                Bn2,
                                np.zeros((2, N // 2 - 2)),
                            ],
                            [
                                np.zeros((N // 2 - 2, 2)),
                                np.eye(N // 2 - 2),
                                np.zeros((N // 2 - 2, 2)),
                                -np.eye(N // 2 - 2),
                            ],
                        ]
                    )

                    S_N = (
                        1
                        / np.sqrt(2)
                        * np.matmul(S_N, block_diag(S_prev, S_prev))
                    )
                    S_prev = S_N
                    i += 1
                return S_prev

        s = slant(min(size, 8))  # /np.sqrt(32)
        f = np.dot(s[i, :size].reshape(-1, 1), s[j, :size].reshape(1, -1))
        return torch.as_tensor(f)

    def forward(self, x):
        in_ = x
        x = F.conv2d(
            x.float(),
            weight=self.filter_bank,
            padding=self.padding,
            stride=self.stride,
            groups=self.input_channels,
        )  # int(self.K/
        x = self.alpha_rooting(x, alpha=self.alpha_root)
        if self.noise is not None:
            x.add_(self.noise.unsqueeze(-1).unsqueeze(-1))
        if self.bn:
            x = F.relu(self.bnorm(x))
        else:
            x = F.relu(x)
        if self.drop:
            x = self.dropout(x)
        if self.use_res:
            x = self.conv(x) + self.shortcut(in_)
        else:
            x = self.conv(x)
        x = F.relu(x)  # ?? do we need this relu layer?
        return x


import torch
import torch.nn as nn
from einops import rearrange

import torch.nn.functional as F

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class Attention(nn.Module):
    def __init__(self, dim, dim_head, heads):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),(q_inp, k_inp, v_inp))
        v = v
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out = self.proj(x).view(b, h, w, c)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class Self_attention_moudle(nn.Module):
    def __init__(self, dim, dim_head, heads, num_blocks=1):
        super(Self_attention_moudle, self).__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, Attention(dim=dim, dim_head=dim_head, heads=heads)),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        b, c, hw = x.size()
        x = rearrange(x, 'b c (h w) -> b c h w', h=int(hw ** 0.5), w=int(hw ** 0.5))
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    """
    def __init__(
            self, in_features,  act_layer=nn.ReLU, norm_layer=nn.LayerNorm, drop=0.5):
        super().__init__()

        self.fc1 = nn.Linear(in_features=in_features, out_features=1, bias=False)
        self.norm = norm_layer(1)
        self.act = act_layer()
        self.fc2 = nn.Softmax(dim=2)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class Positional_Attention_Module(nn.Module):
    def __init__(self, in_channels):
        super(Positional_Attention_Module, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=1, padding=2)
        self.conv6 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=7, stride=1, padding=3)
        self.conv7 = nn.Conv2d(in_channels=in_channels*4, out_channels=in_channels*2, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.mlp1 = ConvMlp(in_features=in_channels)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=1, bias=False),
            nn.Softmax(dim=2)
        )
        self.conv9 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = rearrange(x1, 'b c h w -> b (h w) c')
        x2 = self.conv2(x)
        x2 = rearrange(x2, 'b c h w -> b c (h w)')
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv5(x)
        x6 = self.conv6(x)
        feature1 = torch.einsum('b n i, b j n -> b i j', x1, x2)
        feature1 = self.mlp1(feature1)
        feature2 = torch.cat([x3, x4, x5, x6], dim=1)
        feature2 = self.conv7(feature2)
        feature2 = self.conv8(feature2)
        feature2 = rearrange(feature2, 'b c h w -> b c (h w)')
        out = feature1 * feature2
        b, c, hw = out.size()
        out = rearrange(out, 'b c (h w) -> b c h w', h = int(hw ** 0.5), w = int(hw ** 0.5))
        out = self.conv9(out)
        return out


class GCIE(nn.Module):
    def __init__(self, feature_dim, sram_dim):
        super(GCIE, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=feature_dim, out_channels=sram_dim, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=sram_dim, out_channels=sram_dim, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.Self_attention_moudle = Self_attention_moudle(dim=sram_dim//2,heads=4, dim_head=16)
        self.Positional_Attention_Module = Positional_Attention_Module(in_channels=sram_dim//2)

    def forward(self, x):
        x = self.conv1(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = rearrange(x1, 'b c h w -> b c (h w)')
        x1 = self.long_range_attention_moudle(x1)
        x2 = self.short_range_attention_moudle(x2)
        out = torch.cat([x1, x2], dim=1)
        out = self.conv2(out)
        return out


############################################################################################################################
import torch
import torch.nn as nn


class FFCSE_block(nn.Module):

    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r,
                               kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(
            channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(
            channels // r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))

        x_l = 0 if self.conv_a2l is None else id_l * \
            self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * \
            self.sigmoid(self.conv_a2g(x))
        return x_l, x_g


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()

        # (batch, c, h, w/2+1, 2)
        ffted = torch.rfft(x, signal_ndim=2, normalized=True)
        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)

        output = torch.irfft(ffted, signal_ndim=2,
                             signal_sizes=r_size[2:], normalized=True)

        return output


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s_h = h // split_no
            split_s_w = w // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s_h, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 enable_lfu=True):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        self.bn_l = lnorm(int(out_channels * (1 - ratio_gout)))
        self.bn_g = gnorm(int(out_channels * ratio_gout))

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g

############################################################################
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


# for conv5
class A2CN(nn.Module):
    def __init__(self, cur_channel):
        super(A2CN, self).__init__()
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = nn.Conv2d(cur_channel, cur_channel, kernel_size=1, stride=1, padding=0)
        self.silu = nn.SiLU()
        self.relu = nn.ReLU(True)

        # current conv
        self.cur_b1 = BasicConv2d(cur_channel, cur_channel//4, 3, padding=1, dilation=1)
        self.cur_b2 = BasicConv2d(cur_channel, cur_channel//4, 3, padding=2, dilation=2)
        self.cur_b3 = BasicConv2d(cur_channel, cur_channel//4, 3, padding=3, dilation=3)
        self.cur_b4 = BasicConv2d(cur_channel, cur_channel//4, 3, padding=4, dilation=4)

        self.cur_all = BasicConv2d(cur_channel, cur_channel, 3, padding=1)
        self.cur_all_ca = ChannelAttention(cur_channel)
        self.cur_all_sa = SpatialAttention()

        # previous conv
        self.downsample2 = nn.MaxPool2d(2, stride=2)
        self.pre_sa = SpatialAttention()


    def forward(self, x):
        # current conv
        x_cur_1 = self.cur_b1(x[1])
        x_cur_2 = self.cur_b2(x[1])
        x_cur_3 = self.cur_b3(x[1])
        x_cur_4 = self.cur_b4(x[1])
        x_cur_all = self.cur_all(torch.cat((x_cur_1, x_cur_2, x_cur_3, x_cur_4), 1))
        cur_all_ca = x_cur_all.mul(self.cur_all_ca(x_cur_all))
        cur_all_sa = x_cur_all.mul(self.cur_all_sa(cur_all_ca))

        # previois conv
        x[0] = self.downsample2(x[0])
        pre_sa = x_cur_all.mul(self.pre_sa(x[0]))

        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        x_A2CN=self.conv(self.silu(weight[0] * cur_all_sa + weight[1] * pre_sa + weight[2] * x[1]))

        return x_A2CN

