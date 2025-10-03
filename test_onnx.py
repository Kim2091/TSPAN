import onnxruntime as ort
import numpy as np
import cv2
import os
import argparse
import time
from glob import glob
import math
from datetime import timedelta
from fractions import Fraction

from utils.utils_video import VideoDecoder, VideoEncoder
from utils.logger import logger
from utils import utils_image as util


def detect_model_precision(model_path, session):
    if '_fp16' in model_path.lower():
        return 'fp16'
    elif '_fp32' in model_path.lower():
        return 'fp32'
    
    input_type = session.get_inputs()[0].type
    if 'float16' in input_type:
        return 'fp16'
    
    return 'fp32'


def main():
    parser = argparse.ArgumentParser(description="Test TSPAN ONNX model inference with video")
    parser.add_argument('--model_path', type=str, required=True, help='path to the ONNX model')
    parser.add_argument('--input', type=str, default='input', help='path of input video or directory')
    parser.add_argument('--output', type=str, default='output', help='path of output video or directory')
    parser.add_argument('--depth', type=int, default=8, help='bit depth of outputs')
    parser.add_argument('--suffix', type=str, default=None, help='output filename suffix')
    parser.add_argument('--video', type=str, default=None, help='ffmpeg video codec. if chosen, output video instead of images', 
                        choices=['dnxhd', 'h264_nvenc', 'libx264', 'libx265', '...'])
    parser.add_argument('--crf', type=int, default=11, help='video crf')
    parser.add_argument('--preset', type=str, default='slow', help='video preset')
    parser.add_argument('--fps', type=str, default=None, 
                        help='video framerate (defaults to input video\'s frame rate when processing video)')
    parser.add_argument('--res', type=str, default=None, help='video resolution to scale output to (optional, auto-calculated if not specified)')
    parser.add_argument('--presize', action='store_true', help='resize video before processing')
    parser.add_argument('--providers', type=str, default='CUDAExecutionProvider,CPUExecutionProvider', 
                        help='ONNX Runtime execution providers, comma separated')
    parser.add_argument('--precision', type=str, default=None, choices=['fp16', 'fp32'],
                        help='Model precision (auto-detected if not specified)')
    parser.add_argument('--gui-mode', action='store_true', 
                        help='Output progress in a format optimized for GUI parsing')

    args = parser.parse_args()

    if not args.model_path:
        parser.print_help()
        raise ValueError('Please specify model_path')

    model_path = args.model_path
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    L_path = args.input
    E_path = args.output

    if not L_path or not os.path.exists(L_path):
        logger.error('Error: input path does not exist.')
        return
    
    video_input = False
    if L_path.split('.')[-1].lower() in ['webm','mkv', 'flv', 'vob', 'ogv', 'ogg', 'drc', 'gif', 'gifv', 'mng', 'avi', 'mts', 
                                         'm2ts', 'ts', 'mov', 'qt', 'wmv', 'yuv', 'rm', 'rmvb', 'viv', 'asf', 'amv', 'mp4', 
                                         'm4p', 'm4v', 'mpg', 'mp2', 'mpeg', 'mpe', 'mpv', 'm2v', 'm4v', 'svi', '3gp', '3g2', 
                                         'mxf', 'roq', 'nsv', 'f4v', 'f4p', 'f4a', 'f4b']:
        video_input = True
        if not args.video:
            logger.error('Error: input video requires --video to be set')
            return
    elif os.path.isdir(L_path):
        L_paths = util.get_image_paths(L_path)
    else:
        L_paths = [L_path]

    if args.video and (not E_path or os.path.isdir(E_path)):
        logger.error('Error: output path must be a single video file')
        return

    if not os.path.exists(E_path) and os.path.splitext(E_path)[1] == '':
        util.mkdir(E_path)
    if not args.video and not os.path.isdir(E_path) and os.path.isdir(L_path):
        E_path = os.path.dirname(E_path)
    
    if not model_path.endswith('_fp32.onnx') and not model_path.endswith('_fp16.onnx'):
        if args.precision:
            base_path = os.path.splitext(model_path)[0]
            if base_path.endswith('.onnx'):
                base_path = base_path[:-5]
            model_path = f"{base_path}_{args.precision}.onnx"
        else:
            base_path = os.path.splitext(model_path)[0]
            if base_path.endswith('.onnx'):
                base_path = base_path[:-5]
            
            fp32_path = f"{base_path}_fp32.onnx"
            fp16_path = f"{base_path}_fp16.onnx"
            
            if os.path.exists(fp32_path):
                model_path = fp32_path
            elif os.path.exists(fp16_path):
                model_path = fp16_path
            else:
                logger.error(f'Error: Could not find model at {fp32_path} or {fp16_path}')
                return
    
    logger.info(f"Loading ONNX model from {model_path}")
    
    providers = [p.strip() for p in args.providers.split(',')]
    
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(model_path, sess_options, providers=providers)
    
    model_precision = detect_model_precision(model_path, session)
    logger.info(f"Model precision: {model_precision.upper()}")
    
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_dtype = session.get_inputs()[0].type
    
    logger.info(f"Input dtype: {input_dtype}")
    
    clip_size = input_shape[1] // 3
    
    input_height_required = input_shape[2] if isinstance(input_shape[2], int) and input_shape[2] > 0 else None
    input_width_required = input_shape[3] if isinstance(input_shape[3], int) and input_shape[3] > 0 else None
    
    test_height = input_height_required if input_height_required else 256
    test_width = input_width_required if input_width_required else 256
    
    logger.info(f"Creating test input with shape (1, {clip_size * 3}, {test_height}, {test_width})")
    
    if model_precision == 'fp16':
        test_input = np.zeros((1, clip_size * 3, test_height, test_width), dtype=np.float16)
    else:
        test_input = np.zeros((1, clip_size * 3, test_height, test_width), dtype=np.float32)
    
    test_output = session.run(None, {input_name: test_input})[0]
    logger.info(f"Test output shape: {test_output.shape}")

    if len(test_output.shape) == 5:
        scale = test_output.shape[3] // test_height
    elif len(test_output.shape) == 4:
        scale = test_output.shape[2] // test_height
    
    logger.info(f"Model: {model_name}")
    logger.info(f"Clip size: {clip_size}")
    logger.info(f"Scale: {scale}x")
    logger.info(f"Output shape from test: {test_output.shape}")

    n_channels = 3
    
    if video_input:
        video_decoder = VideoDecoder(L_path, options={'r': '24000/1001'})
        img_count = len(video_decoder)
        video_decoder.start()
        
        first_frame = video_decoder.get_frame()
        input_height, input_width = first_frame.shape[:2]
        video_decoder.stop()
        
        import av
        with av.open(L_path) as container:
            input_fps = container.streams.video[0].average_rate
        
        video_decoder = VideoDecoder(L_path, options={'r': str(input_fps)})
        video_decoder.start()
    else:
        if len(L_paths) > 0:
            first_img = util.imread_uint(L_paths[0], n_channels=n_channels)
            input_height, input_width = first_img.shape[:2]
        else:
            logger.error('Error: no input images found.')
            return

    if args.res is None:
        if args.presize:
            output_width = input_width
            output_height = input_height
        else:
            output_width = input_width * scale
            output_height = input_height * scale
        output_res = f"{output_width}:{output_height}"
    else:
        output_res = args.res

    logger.info(f"Input resolution: {input_width}x{input_height}")
    logger.info(f"Output resolution: {output_res}")

    input_window = []
    image_names = []
    total_time = 0
    end_of_video = False
    video_encoder = None
    
    try:
        if args.video:
            if args.fps is None and video_input:
                fps = input_fps
            elif args.fps is None:
                fps = Fraction(24000, 1001)
            elif '/' in args.fps:
                fps = Fraction(*map(int, args.fps.split('/')))
            elif '.' in args.fps:
                fps = float(args.fps)
            else:
                fps = int(args.fps)

            codec_options = {
                'crf': str(args.crf),
                'preset': args.preset,
            }
            video_encoder = VideoEncoder(
                E_path,
                int(output_res.split(':')[0]),
                int(output_res.split(':')[1]),
                fps=fps,
                codec=args.video,
                options=codec_options,
                input_depth=args.depth,
            )
            video_encoder.start()

        if args.suffix:
            suffix = f"{scale}x_{args.suffix}"
        else:
            suffix = f"{model_name}" if f"{scale}x_" in model_name else f"{scale}x_{model_name}"

        idx = 0
        while True:
            start_time = time.time()
            
            if video_input:
                img_L = video_decoder.get_frame()
            elif len(L_paths) == 0:
                img_L = None
            else:
                img_L = L_paths.pop(0)
                img_name, ext = os.path.splitext(os.path.basename(img_L))
                img_L = util.imread_uint(img_L, n_channels=n_channels)
                image_names += [img_name]
            
            if img_L is None and not end_of_video:
                img_count = idx + clip_size // 2
                end_of_video = True
                input_window += input_window[clip_size//2-1:-1][::-1]
            elif not end_of_video:
                if args.presize:
                    img_L = cv2.resize(img_L, (int(output_res.split(':')[0])//scale, int(output_res.split(':')[1])//scale), interpolation=cv2.INTER_CUBIC)
                
                img_L_np = img_L.astype(np.float32) / 255.0
                img_L_np = np.transpose(img_L_np, (2, 0, 1))
                input_window += [img_L_np]

            if len(input_window) < clip_size and end_of_video:
                break
            elif len(input_window) < clip_size // 2 + 1:
                continue
            elif len(input_window) == clip_size // 2 + 1:
                input_window = input_window[1:][::-1] + input_window

            if len(input_window) < clip_size:
                continue

            window_np = np.stack(input_window[:clip_size], axis=0)
            
            curr_height, curr_width = window_np.shape[2], window_np.shape[3]
            pad_height = -(-curr_height // 64) * 64 + 64
            pad_width = -(-curr_width // 64) * 64 + 64
            pad_h = (pad_height - curr_height) // 2
            pad_w = (pad_width - curr_width) // 2

            padded_window = np.zeros((1, clip_size, 3, pad_height, pad_width), dtype=np.float32)
            
            for i in range(clip_size):
                frame = window_np[i]
                padded_window[0, i] = np.pad(frame,
                                           ((0, 0),
                                            (pad_h, pad_height - curr_height - pad_h),
                                            (pad_w, pad_width - curr_width - pad_w)),
                                           mode='reflect')

            padded_window = padded_window.reshape(1, clip_size * 3, pad_height, pad_width)

            if model_precision == 'fp16':
                padded_window = padded_window.astype(np.float16)
            else:
                padded_window = padded_window.astype(np.float32)

            outputs = session.run(None, {input_name: padded_window})

            if idx == 0:
                logger.info(f"Original input shape: {window_np.shape}")
                logger.info(f"Output shape before unpad: {outputs[0].shape}")
                logger.info(f"Final output shape: ({curr_height * scale}, {curr_width * scale})")

            img_E_np = outputs[0][0]
            
            if model_precision == 'fp16':
                img_E_np = img_E_np.astype(np.float32)

            input_window.pop(0)

            if len(img_E_np.shape) == 3:
                h_start = (img_E_np.shape[1] - curr_height * scale) // 2
                h_end = h_start + curr_height * scale
                w_start = (img_E_np.shape[2] - curr_width * scale) // 2
                w_end = w_start + curr_width * scale
                img_E_np = img_E_np[:, h_start:h_end, w_start:w_end]

            img_E_np = np.clip(img_E_np, 0, 1)
            if args.depth == 8:
                img_E = (img_E_np * 255).astype(np.uint8)
            else:
                img_E = (img_E_np * ((1 << args.depth) - 1)).astype(np.uint16)
            
            img_E = np.transpose(img_E, (1, 2, 0))

            if args.video:
                img_E = cv2.resize(img_E, (int(output_res.split(':')[0]), int(output_res.split(':')[1])), interpolation=cv2.INTER_CUBIC)
                video_encoder.add_frame(img_E)
            elif os.path.isdir(E_path):
                util.imsave(img_E, os.path.join(E_path, f'{image_names.pop(0)}_{suffix}.png'))
            else:
                util.imsave(img_E, E_path)

            end_time = time.time()
            time_taken = (end_time - start_time) * 1000
            total_time += time_taken
            
            idx += 1
            time_remaining = ((total_time / idx) * (img_count - idx)) / 1000

            if args.gui_mode:
                print(f'PROGRESS:{idx}/{img_count}|FPS:{1000/time_taken:.2f}', flush=True)
            else:       
                print(f'{idx}/{img_count}   fps: {1000/time_taken:.2f}  frame time: {time_taken:.2f}ms   time remaining: {math.trunc(time_remaining/3600)}h{math.trunc((time_remaining/60)%60)}m{math.trunc(time_remaining%60)}s ', end='\r')
                
    except KeyboardInterrupt:
        logger.info("\nCaught KeyboardInterrupt, ending gracefully")
    except Exception as e:
        logger.error(f"\nError: {str(e)}")
    finally:
        if video_input and video_decoder is not None:
            try:
                video_decoder.stop()
            except:
                pass

        if video_encoder is not None:
            try:
                video_encoder.stop()
                video_encoder.join(timeout=5)
                
                if hasattr(video_encoder, 'output_container') and video_encoder.output_container:
                    video_encoder.output_container.close()
                
                if idx > 0:
                    logger.info(f"Saved video to {E_path}")
                    output_dir = os.path.dirname(os.path.abspath(E_path))
                    print(f"\033]8;;file://{output_dir}\033\\Click to open output directory\033]8;;\033\\")
            except Exception as e:
                logger.error(f"Error while closing video encoder: {e}")

        if idx > 0:
            average_fps = idx / (total_time / 1000)
            logger.info(f'Processed {idx} images in {timedelta(milliseconds=total_time)}, average {average_fps:.2f} FPS')
        os._exit(0)


if __name__ == '__main__':
    main()
