
import vapoursynth as vs
core = vs.core

# load clip
source_path = "path/to/video.mkv"
clip = core.bs.VideoSource(source=source_path, cachemode=3)

# convert to RGBS for vsmlrt. Use RGBH if running an FP16 model
clip = core.resize.Bilinear(clip, format=vs.RGBS, matrix_in_s="470bg")

# inference with 5 input frames
m2 = clip[:2] + clip[:-2]   # shift -2
m1 = clip[:1] + clip[:-1]   # shift -1
p1 = clip[1:] + clip[-1:]   # shift +1
p2 = clip[2:] + clip[-2:]   # shift +2
clip = core.trt.Model([m2, m1, clip, p1, p2], engine_path="path/to/tspan.engine", num_streams=1)

# convert to output format
clip = core.resize.Bilinear(clip, format=vs.YUV420P8, matrix_s="709")

# set output
clip.set_output()

















