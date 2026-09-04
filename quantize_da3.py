import os
import sys
import time
import numpy as np
from PIL import Image
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType, CalibrationMethod

class SampleDataReader(CalibrationDataReader):
    def __init__(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize((504, 504), Image.Resampling.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))
        inp = np.expand_dims(arr, 0).astype(np.float32)
        
        self.data = iter([{'image': inp}])

    def get_next(self):
        return next(self.data, None)

def main():
    model_in = '/home/ubuntu/da3-npu/models/model_fp32_nomatconv.onnx'
    model_out = '/home/ubuntu/da3-npu/models/model_da3_qdq.onnx'
    sample_img = '/home/ubuntu/da3-npu/data/sample.png'

    nodes_to_exclude = [
        '/model/head/output_conv2/output_conv2.2/Conv_mm',
        '/model/head/sky_output_conv2/sky_output_conv2.2/Conv_mm'
    ]
    print(f'Excluding nodes from quantization: {nodes_to_exclude}')

    ops = ['Conv', 'ConvTranspose', 'Resize', 'MatMul', 'Relu']
    print(f'Quantizing op types: {ops}')

    t0 = time.time()
    reader = SampleDataReader(sample_img)

    quantize_static(
        model_input=model_in,
        model_output=model_out,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=ops,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        nodes_to_exclude=nodes_to_exclude,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=False,
    )

    t1 = time.time()
    print(f'Quantization complete in {t1 - t0:.2f}s! Saved to {model_out}')
    print(f'Output file size: {os.path.getsize(model_out) / (1024*1024):.2f} MB')

if __name__ == '__main__':
    main()
