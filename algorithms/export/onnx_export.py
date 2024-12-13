import torch

def export_to_torchscript(model, input_example, output_path):
    """
    Exports a PyTorch model to TorchScript.
    
    Args:
        model (torch.nn.Module): The trained PyTorch model.
        input_example (torch.Tensor): Example input for tracing.
        output_path (str): File path to save the TorchScript model.
    
    Returns:
        None
    """
    try:
        model.eval()  # Ensure the model is in evaluation mode
        scripted_model = torch.jit.trace(model, input_example)
        torch.jit.save(scripted_model, output_path)
        print(f"TorchScript model successfully saved to {output_path}")
    except Exception as e:
        print(f"Error exporting model to TorchScript: {e}")
        raise

#from marl.export.torchscript_export import export_to_torchscript
#import torch

## Assuming `trained_model` is your PyTorch model and `example_input` is a dummy input tensor
#export_to_torchscript(trained_model, example_input=torch.randn(1, 10), output_path="models/my_model_scripted.pt")