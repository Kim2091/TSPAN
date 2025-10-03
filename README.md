# TSPAN
TSPAN (Temporal SPAN) is a custom arch based on the [original SPAN architecture](https://github.com/hongyuanyu/SPAN). This modification aims to provide efficient and accessible temporal video upscaling. It supports PyTorch, ONNX, and TensorRT!

## Getting Started

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Kim2091/TSPAN
    ```

2.  **Install PyTorch with CUDA**:
    Follow the instructions at [pytorch.org](https://pytorch.org/get-started/locally/).

3.  **Install required packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can use TSPAN through the graphical user interface (GUI) or the command line.

## TensorRT

For high-performance inference, refer to the [TensorRT guide](tensorrt/README.md).

### GUI Usage

For an easy-to-use experience with PyTorch or ONNX models, launch the GUI:

```bash
python vsr_gui.py
```

<img width="602" height="698" alt="image" src="https://github.com/user-attachments/assets/744fd695-3fe8-4dc7-b52c-f3bca423e13c" />


### Command-Line Usage

For more advanced control, you can use the command-line scripts.

**Video upscaling (PyTorch)**:
```bash
python test_vsr.py --model_path pretrained_models/tspan.pth --input path/to/video.mp4 --output path/to/output.mp4
```

**ONNX Video upscaling**:
```bash
python test_onnx.py --model_path model.onnx --input path/to/video.mp4 --output path/to/output.mp4
```

Key arguments for `test_vsr.py` and `test_onnx.py`:
-   `--video_codec`: Specify the video codec (e.g., `libx264`, `libx265`).
-   `--crf`: Set the Constant Rate Factor for quality (for `libx264`/`libx265`).
-   `--providers`: (ONNX only) Set ONNX Runtime execution providers.

## Tools

Utility scripts are located in the `tools/` directory.

**Convert PyTorch model to ONNX**:
```bash
python tools/convert_to_onnx.py --model pretrained_models/model.pth --output model.onnx
```
-   `--dynamic`: Create a model that supports various input sizes.
-   `--fp16`: Convert the model to FP16 for a speed boost.

**Convert FP32 model to FP16**:
```bash
python tools/pth_fp32_to_fp16.py --model path/to/tspan.pth --output path/to/output.pth
```

## Credits (thanks all!)
- Uses [SCUNet](https://github.com/aaf6aa/SCUNet)'s respository as a base
- Uses a modified version of [SPAN](https://github.com/hongyuanyu/SPAN)'s architecture
- Uses the additions made in [both](https://github.com/aaf6aa/SCUNet) [TSCUNet](https://github.com/Kim2091/TSCUNet) repositories




