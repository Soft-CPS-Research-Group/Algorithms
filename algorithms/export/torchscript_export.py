import torch
import os

def export_to_onnx(model, input_example, output_path, opset_version=11):
    """
    Exports a PyTorch model to ONNX format.
    
    Args:
        model (torch.nn.Module): The trained PyTorch model.
        input_example (torch.Tensor): Example input for tracing.
        output_path (str): File path to save the ONNX model.
        opset_version (int): ONNX opset version. Default is 11.
    
    Returns:
        None
    """
    try:
        model.eval()  # Ensure the model is in evaluation mode
        if not output_path.endswith(".onnx"):
            output_path += ".onnx"
        torch.onnx.export(
            model,
            input_example,
            output_path,
            export_params=True,  # Store the trained weights
            opset_version=opset_version,
            input_names=["input"],  # Optional: name the input tensor
            output_names=["output"],  # Optional: name the output tensor
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}  # Dynamic batch size
        )
        print(f"ONNX model successfully saved to {output_path}")
    except Exception as e:
        print(f"Error exporting model to ONNX: {e}")
        raise

#from marl.export.onnx_export import export_to_onnx
#import torch

## Assuming `trained_model` is your PyTorch model and `example_input` is a dummy input tensor
# export_to_onnx(trained_model, input_example=torch.randn(1, 10), output_path="models/my_model.onnx")

