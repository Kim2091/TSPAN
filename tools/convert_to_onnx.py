import torch
import torch.onnx
import argparse
import os
import numpy as np
import onnx
from onnxconverter_common import float16

# Make sure the new network is imported
from models.network_tspan import TSPAN

class TemporalSPANExportWrapper(torch.nn.Module):
    """
    A simple wrapper for the TemporalSPAN model (named TSPAN) to handle
    the input shape transformation required for ONNX export. The rest of the
    script expects a 4D tensor, while the model's forward pass expects a 5D tensor.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Get clip_size from the model instance for the reshape operation
        self.clip_size = model.clip_size
        # The number of channels per frame is fixed (e.g., RGB)
        self.channels_per_frame = 3

    def forward(self, x):
        """
        Takes a 4D tensor and reshapes it for the model.
        Args:
            x: Input tensor with shape (batch, clip_size * channels, height, width)
        """
        # Get dynamic shape info from the input tensor
        b, _, h, w = x.shape
        
        # Reshape to the 5D format the model's forward() method expects
        # (batch, clip_size, channels, height, width)
        reshaped_x = x.view(b, self.clip_size, self.channels_per_frame, h, w)
        
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
        np.testing.assert_allclose(torch_output, onnx_output, rtol=rtol, atol=atol)
        logger.info("✓ ONNX output verified successfully against PyTorch output.")
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

def convert_model_to_onnx(model_path, onnx_path, input_shape, dynamic=False, verify=True, fp16=False):
    """
    Convert a TemporalSPAN (TSPAN) PyTorch model to ONNX format.
    """
    logger.info(f"Loading PyTorch model from: {model_path}")
    device = torch.device('cpu')
    
    # Load model state dict and initialize the model
    state_dict = torch.load(model_path, map_location=device)
    model = TSPAN(state=state_dict)
    model.eval()
    model = model.to(device)
    
    logger.info(f"Model Info: clip_size={model.clip_size}, scale={model.scale}x")
    
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
    parser.add_argument("--dynamic", action="store_true", help="Export with dynamic axes for batch, height, and width")
    parser.add_argument("--no-verify", action="store_true", help="Skip ONNX output verification against PyTorch")
    parser.add_argument("--fp16", action="store_true", help="Also create an FP16 version of the model")
    args = parser.parse_args()
    
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.model))[0]
        args.output = f"{base_name}.onnx"
    
    # Temporarily load the model to get its clip_size
    logger.info("Loading model to determine clip_size...")
    temp_state = torch.load(args.model, map_location='cpu')
    clip_size = TSPAN(state=temp_state).clip_size
    del temp_state
    
    # Define the 4D input shape the script and ONNX model will use
    input_shape = (args.batch, clip_size * 3, args.height, args.width)
    
    convert_model_to_onnx(
        args.model, 
        args.output,
        input_shape,
        args.dynamic,
        not args.no_verify,
        args.fp16
    )