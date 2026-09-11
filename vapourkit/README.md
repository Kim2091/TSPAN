# Vapourkit filter

Copy `TSPAN TensorRT.vkfilter` into Vapourkit's `include/plugins/plugin_filters/`
folder. Restart Vapourkit, add the filter, and select your TensorRT engine file.
The filter operates on the current clip; it does not load a source or set an output.

Requires VapourSynth, the vs-mlrt TensorRT plugin (`core.trt.Model`), and a compatible
TSPAN engine built for five RGB frames at the current input resolution. Match
`fp16` to the engine's **I/O bindings**, not merely its internal compute precision.
Models and engines are supplied separately; see [the TensorRT guide](../tensorrt/README.md).

Input FPS and frame count are preserved. Temporal neighbours repeat the first/last
frame at clip boundaries. Output retains the input pixel format at the model's
output resolution. Vapourkit's project matrix supplies the fallback for YUV conversion.

From the repository root, run `python package-vapoursynth.py` to create
`dist/TSPAN-vapoursynth.zip`, containing the example script, filter, guides and license.
Include this filter whenever distributing the associated VapourSynth script.
