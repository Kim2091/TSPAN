# TensorRT Guide

This guide helps you use TSPAN with TensorRT for accelerated video upscaling.

## Setup Process

1. Set up VapourSynth following [pifroggi's guide](https://github.com/pifroggi/vapoursynth-stuff/blob/main/docs/vapoursynth-portable-setup-tutorial.md)
2. Download and extract `vsmlrt-windows-x64-tensorrt.[version].7z` from [vs-mlrt releases](https://github.com/AmusementClub/vs-mlrt/releases) to your `vs-plugins` directory
3. Get the model:
   - Download pre-converted ONNX from [releases](https://github.com/Kim2091/Kim2091-Models/releases), or
   - Convert your own using `convert_to_onnx.py` (see script for detailed options)

## Usage

1. Build TensorRT engine using `trtexec`:

    FP32:
    ```bash
    trtexec --onnx="tspan_fp32.onnx" --optShapes=input:1x15x720x1280 --saveEngine=tspan_fp32.engine --builderOptimizationLevel=5 --useCudaGraph --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT
    ```
    
    FP16:
    ```bash
    trtexec --onnx="tspan_fp16.onnx" --fp16 --optShapes=input:1x15x720x1280 --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --saveEngine=tspan_fp16.engine --builderOptimizationLevel=5 --useCudaGraph --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT
    ```

2. Copy `vapoursynth_script.py` to your VapourSynth directory, then configure it with your video path and engine path

3. Open a Command Prompt window (NOT POWERSHELL) in your VapourSynth directory, then run a command like this. Customize the encoder settings as you wish:
```bash
vspipe -c y4m ".\vapoursynth_script.vpy" - | ffmpeg -i - -c:v hevc_nvenc -qp 0 -preset p5 -tune lossless "output.mkv"
```
