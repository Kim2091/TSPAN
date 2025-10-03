import argparse
import math
import os.path
import torch

from datetime import timedelta

from utils import utils_image as util
from utils.logger import logger
from utils.utils_video import VideoDecoder, VideoEncoder
from models.temporal_span_arch import TemporalSPAN as net

if not torch.cuda.is_available():
    logger.error('CUDA is not available. Exiting...')
    exit()

default_device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

if torch.cuda.is_bf16_supported():
    default_dtype = torch.bfloat16
else:
    props = torch.cuda.get_device_properties(default_device)
    # fp16 supported at compute 5.3 and above
    if props.major > 5 or (props.major == 5 and props.minor >= 3):
        default_dtype = torch.float16
    else:
        default_dtype = torch.float32

def main():
    n_channels = 3

    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='path to the model')
    parser.add_argument('--input', type=str, required=True, help='path of input video')
    parser.add_argument('--output', type=str, required=True, help='path of output video')
    parser.add_argument('--video_codec', type=str, default='libx264', help='ffmpeg video codec', choices=['dnxhd', 'libx264', 'libx265'])
    parser.add_argument('--crf', type=int, default=15, help='Constant Rate Factor (CRF) for x264/x265 codecs')
    parser.add_argument('--gui-mode', action='store_true', help='Output progress in a format optimized for GUI parsing')

    args = parser.parse_args()

    model_path = args.model_path
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    # ----------------------------------------
    # L_path, E_path
    # ----------------------------------------
    L_path = args.input
    E_path = args.output

    if not os.path.exists(L_path):
        logger.error('Error: input path does not exist.')
        return
    
    if os.path.isdir(E_path):
        logger.error('Error: output path must be a file, not a directory.')
        return

    # ----------------------------------------
    # load model
    # ----------------------------------------
    # Load checkpoint
    checkpoint = torch.load(model_path)
    state_dict = checkpoint.get('params_ema', checkpoint)

    # Infer scale from checkpoint
    if 'upsampler.0.weight' in state_dict:
        upsampler_weight = state_dict['upsampler.0.weight']
        scale = int((upsampler_weight.shape[0] / 3) ** 0.5)
    else:
        scale = 4  # default

    num_frames = 5  # default

    clip_size = num_frames

    # Create model with inferred parameters
    model = net(
        num_in_ch=3,
        num_out_ch=3,
        num_frames=num_frames,
        feature_channels=48,
        upscale=scale,
        bias=True,
        history_channels=12,
    )

    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(default_device)    
    
    # warmup
    input_shape = (1, clip_size, 3, 540, 720)
    dummy_input = torch.randn(input_shape).to(default_device, dtype=default_dtype)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=default_dtype):
        _ = model(dummy_input)

    logger.info(f'Model path: {model_path}')
    logger.info(f'model_name: {model_name}')
    logger.info(f'scale: {scale}x')
    logger.info(f'Input video: {L_path}')

    video_decoder = VideoDecoder(L_path)
    img_count = len(video_decoder)
    video_decoder.start()
    
    input_height, input_width = video_decoder.height, video_decoder.width
    input_fps = video_decoder.fps

    output_width = input_width * scale
    output_height = input_height * scale
    output_res = f"{output_width}:{output_height}"

    input_window = []

    total_time = 0
    end_of_video = False
    
    idx = 0
    video_encoder = None
    try:
        encoder_options = {}
        if args.video_codec in ['libx264', 'libx265']:
            encoder_options['crf'] = str(args.crf)

        video_encoder = VideoEncoder(
            E_path,
            output_width,
            output_height,
            fps=input_fps,
            codec=args.video_codec,
            options=encoder_options
        )
        video_encoder.start()

        while True:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            img_L = video_decoder.get_frame()
            
            if img_L is None:
                if not end_of_video:
                    img_count = idx + clip_size // 2
                    end_of_video = True
                    # Reflect pad the end of the window
                    input_window.extend(input_window[clip_size//2-1:-1][::-1])
            else:
                img_L_t = util.uint2tensor4(img_L).to(default_device, dtype=default_dtype)
                input_window.append(img_L_t)

            if len(input_window) < clip_size and end_of_video:
                break
            elif len(input_window) < clip_size // 2 + 1:
                continue
            elif len(input_window) == clip_size // 2 + 1:
                # Reflect pad the beginning of the window
                input_window = input_window[1:][::-1] + input_window

            window = torch.stack(input_window[:clip_size], dim=1)
            
            with torch.amp.autocast('cuda', dtype=default_dtype):
                img_E = model(window)
            
            del window
            input_window.pop(0)

            img_E = util.tensor2uint(img_E, 8)
            
            video_encoder.add_frame(img_E)

            end.record()
            torch.cuda.synchronize()

            idx += 1
            time_taken = start.elapsed_time(end)
            total_time += time_taken
            time_remaining = ((total_time / idx) * (img_count - (idx+1))) / 1000

            if args.gui_mode:
                print(f'PROGRESS:{idx}/{img_count}|FPS:{1000/time_taken:.2f}', flush=True)
            else:
                print(f'{idx}/{img_count}   fps: {1000/time_taken:.2f}  frame time: {time_taken:.2f}ms   time remaining: {math.trunc(time_remaining/3600)}h{math.trunc((time_remaining/60)%60)}m{math.trunc(time_remaining%60)}s ', end='\r')

    except KeyboardInterrupt:
        logger.info("\nCaught KeyboardInterrupt, ending gracefully")
    except Exception as e:
        logger.error(f"\nAn error occurred: {e}")
    finally:
        if video_decoder:
            video_decoder.stop()
        if video_encoder:
            video_encoder.stop()
            video_encoder.join(timeout=5)
            if idx > 0:
                logger.info(f"\nSaved video to {E_path}")
                output_dir = os.path.dirname(os.path.abspath(E_path))
                print(f"\033]8;;file://{output_dir}\033\\Click to open output directory\033]8;;\033\\")

        if idx > 0:
            average_fps = idx / (total_time / 1000)  # Convert ms to seconds
            logger.info(f'Processed {idx} images in {timedelta(milliseconds=total_time)}, average {average_fps:.2f} FPS')        
        os._exit(0)

if __name__ == '__main__':
    main()