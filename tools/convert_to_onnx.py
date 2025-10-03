import torch
import torch.onnx
import argparse
import os
import sys
import numpy as np
import onnx
from onnxconverter_common import float16
import logging

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Import the correct model architecture
from models.temporal_span_arch import TemporalSPAN

class TemporalSPANExportWrapper(torch.nn.Module):
    """
    A simple wrapper for the TemporalSPAN model to handle
    the input shape transformation required for ONNX export. The rest of the
    script expects a 4D tensor, while the model's forward pass expects a 5D tensor.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Get num_frames from the model instance for the reshape operation
        self.num_frames = model.num_frames
        # The number of channels per frame is fixed (e.g., RGB)
        self.channels_per_frame = model.in_channels

    def forward(self, x):
        """
        Takes a 4D tensor and reshapes it for the model.
        Args:
            x: Input tensor with shape (batch, num_frames * channels, height, width)
        """
        # Get dynamic shape info from the input tensor
        b, _, h, w = x.shape
        
        # Reshape to the 5D format the model's forward() method expects
        # (batch, num_frames, channels, height, width)
        reshaped_x = x.view(b, self.num_frames, self.channels_per_frame, h, w)
        
        # Call the original model's forward pass with the correctly shaped tensor
        return self.model(reshaped_x)

def verify_onnx_output(model, onnx_path, test_input, rtol=1e-3, atol=1e-4):
    """
    Verify ONNX model output against PyTorch model output.
    This function remains largely the same but is now called with the wrapper.
    """
    try:
        import onnxruntime as ort
        
        logger.info("\nVerifying ONNX model...")

        # Get PyTorch output using the wrapper
        model.eval()
        with torch.inference_mode():
            torch_output = model(test_input).cpu().numpy()

        # Load and check the ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Create ONNX Runtime session
        ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        
        # Prepare input for ONNX Runtime
        ort_inputs = {ort_session.get_inputs()[0].name: test_input.cpu().numpy()}
        
        # Run ONNX model
        onnx_output = ort_session.run(None, ort_inputs)[0]

        # Compare outputs
        logger.info(f"PyTorch output shape: {torch_output.shape}")
        logger.info(f"ONNX output shape:    {onnx_output.shape}")
        
        # Calculate detailed difference metrics
        abs_diff = np.abs(torch_output - onnx_output)
        rel_diff = abs_diff / (np.abs(torch_output) + 1e-8)
        
        logger.info("\n=== Difference Metrics ===")
        logger.info(f"Absolute difference:")
        logger.info(f"  Mean: {abs_diff.mean():.6e}")
        logger.info(f"  Max:  {abs_diff.max():.6e}")
        logger.info(f"  Min:  {abs_diff.min():.6e}")
        
        logger.info(f"\nRelative (percentage) difference:")
        logger.info(f"  Mean: {rel_diff.mean() * 100:.4f}%")
        logger.info(f"  Max:  {rel_diff.max() * 100:.4f}%")
        logger.info(f"  Min:  {rel_diff.min() * 100:.4f}%")
        
        logger.info(f"\nOutput value ranges:")
        logger.info(f"  PyTorch - Min: {torch_output.min():.6f}, Max: {torch_output.max():.6f}")
        logger.info(f"  ONNX    - Min: {onnx_output.min():.6f}, Max: {onnx_output.max():.6f}")
        
        # Perform assertion
        np.testing.assert_allclose(torch_output, onnx_output, rtol=rtol, atol=atol)
        logger.info("\n✓ ONNX output verified successfully against PyTorch output.")
        logger.info(f"  (within rtol={rtol}, atol={atol})")
        return True
            
    except ImportError:
        logger.warning("⚠ ONNX Runtime not installed. Skipping verification.")
        return False
    except Exception as e:
        logger.error(f"❌ Error during ONNX verification: {str(e)}")
        return False

def convert_to_fp16(model_path, output_path=None):
    """Convert an ONNX model to FP16 precision."""
    if output_path is None:
        base_path = model_path.replace('_fp32.onnx', '')
        output_path = f"{base_path}_fp16.onnx"
    
    logger.info(f"\nConverting model to FP16...")
    logger.info(f"Loading ONNX model from: {model_path}")
    
    try:
        onnx_model = onnx.load(model_path)
        onnx_model_fp16 = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
        onnx.save(onnx_model_fp16, output_path)
        logger.info(f"Successfully saved FP16 model to: {output_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Error during FP16 conversion: {e}")
        return False

def load_model_from_state(state_dict, num_in_ch=3, num_out_ch=3, num_frames=5, 
                          feature_channels=48, upscale=4, bias=True, history_channels=12):
    """
    Load a TemporalSPAN model from a state dict.
    
    Args:
        state_dict: The state dictionary to load
        num_in_ch: Number of input channels (default: 3 for RGB)
        num_out_ch: Number of output channels (default: 3 for RGB)
        num_frames: Number of frames in the input sequence
        feature_channels: Number of feature channels
        upscale: Upscaling factor
        bias: Whether to use bias in conv layers
        history_channels: Number of history channels
    
    Returns:
        Loaded TemporalSPAN model
    """
    # Try to infer parameters from state dict if they exist
    if 'params' in state_dict:
        params = state_dict['params']
        num_in_ch = params.get('num_in_ch', num_in_ch)
        num_out_ch = params.get('num_out_ch', num_out_ch)
        num_frames = params.get('num_frames', num_frames)
        feature_channels = params.get('feature_channels', feature_channels)
        upscale = params.get('upscale', upscale)
        bias = params.get('bias', bias)
        history_channels = params.get('history_channels', history_channels)
        state_dict = state_dict['params_ema'] if 'params_ema' in state_dict else state_dict['params']
    elif 'params_ema' in state_dict:
        # Use params_ema if available
        state_dict = state_dict['params_ema']
    
    # Infer scale from upsampler weight shape if available
    if 'upsampler.0.weight' in state_dict:
        upsampler_weight = state_dict['upsampler.0.weight']
        # upsampler output channels = num_out_ch * (scale^2)
        detected_scale = int((upsampler_weight.shape[0] / num_out_ch) ** 0.5)
        logger.info(f"Detected upscale factor from model weights: {detected_scale}x")
        upscale = detected_scale
    
    num_frames = 5
    feature_channels = 32
    
    # Create the model with the correct parameters
    model = TemporalSPAN(
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        num_frames=num_frames,
        feature_channels=feature_channels,
        upscale=upscale,
        bias=bias,
        history_channels=history_channels
    )
    
    # Load the state dict
    model.load_state_dict(state_dict, strict=False)
    
    return model

def convert_model_to_onnx(model_path, onnx_path, input_shape, dynamic=False, verify=True, fp16=False,
                         num_in_ch=3, num_out_ch=3, num_frames=5, feature_channels=48, 
                         upscale=4, bias=True, history_channels=12):
    """
    Convert a TemporalSPAN PyTorch model to ONNX format.
    """
    logger.info(f"Loading PyTorch model from: {model_path}")
    device = torch.device('cpu')
    
    # Load model state dict and initialize the model
    state_dict = torch.load(model_path, map_location=device)
    model = load_model_from_state(
        state_dict, num_in_ch, num_out_ch, num_frames, 
        feature_channels, upscale, bias, history_channels
    )
    model.eval()
    model = model.to(device)
    
    logger.info(f"Model Info: num_frames={model.num_frames}, upscale={upscale}x")
    
    # Create the export wrapper around the loaded model
    export_model = TemporalSPANExportWrapper(model)
    export_model.eval()

    # Create dummy input tensor
    dummy_input = torch.randn(*input_shape, dtype=torch.float32, device=device)
    logger.info(f"Using input shape for export: {input_shape}")
    
    dynamic_axes = None
    if dynamic:
        logger.info("Using dynamic axes for batch, height, and width.")
        dynamic_axes = {
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'out_height', 3: 'out_width'}
        }
    
    # Define output paths for FP32 and FP16 models
    base_path = os.path.splitext(onnx_path)[0]
    fp32_path = f"{base_path}_fp32.onnx"

    logger.info(f"\nExporting model to ONNX (FP32): {fp32_path}")
    try:
        torch.onnx.export(
            export_model,              
            dummy_input,               
            fp32_path,
            export_params=True,        
            opset_version=17,          
            do_constant_folding=True,  
            input_names=['input'],     
            output_names=['output'],   
            dynamic_axes=dynamic_axes,
        )
        logger.info(f"Model successfully exported to {fp32_path}")

        if verify:
            verify_onnx_output(export_model, fp32_path, dummy_input)
        
        if fp16:
            convert_to_fp16(fp32_path)
            
    except Exception as e:
        logger.error(f"❌ Error during ONNX export: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TemporalSPAN model to ONNX")
    parser.add_argument("--model", type=str, required=True, help="Path to the PyTorch model (.pth file)")
    parser.add_argument("--output", type=str, help="Base output path for ONNX model. '_fp32.onnx' and '_fp16.onnx' will be appended.")
    parser.add_argument("--height", type=int, default=256, help="Input height for dummy tensor")
    parser.add_argument("--width", type=int, default=256, help="Input width for dummy tensor")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for dummy tensor")
    parser.add_argument("--num-frames", type=int, default=5, help="Number of frames in the input sequence")
    parser.add_argument("--num-in-ch", type=int, default=3, help="Number of input channels")
    parser.add_argument("--num-out-ch", type=int, default=3, help="Number of output channels")
    parser.add_argument("--feature-channels", type=int, default=48, help="Number of feature channels")
    parser.add_argument("--upscale", type=int, default=4, help="Upscaling factor")
    parser.add_argument("--history-channels", type=int, default=12, help="Number of history channels")
    parser.add_argument("--dynamic", action="store_true", help="Export with dynamic axes for batch, height, and width")
    parser.add_argument("--no-verify", action="store_true", help="Skip ONNX output verification against PyTorch")
    parser.add_argument("--fp16", action="store_true", help="Also create an FP16 version of the model")
    args = parser.parse_args()
    
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.model))[0]
        args.output = f"{base_name}.onnx"
    
    # Define the 4D input shape the script and ONNX model will use
    input_shape = (args.batch, args.num_frames * args.num_in_ch, args.height, args.width)
    
    convert_model_to_onnx(
        args.model, 
        args.output,
        input_shape,
        args.dynamic,
        not args.no_verify,
        args.fp16,
        args.num_in_ch,
        args.num_out_ch,
        args.num_frames,
        args.feature_channels,
        args.upscale,
        True,  # bias
        args.history_channels
    )