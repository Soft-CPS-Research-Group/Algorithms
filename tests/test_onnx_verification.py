"""
Verification test for ONNX export and inference.
This test ensures the ONNX model produces identical outputs to PyTorch.
"""
import pytest
import torch
import numpy as np
import tempfile
import os

from algorithms.utils.transformer_networks import TransformerConfig, TransformerActor
from algorithms.utils.tokenizer import TokenizerConfig, ObservationTokenizer
from algorithms.export.transformer_onnx_export import (
    export_transformer_actor_to_onnx,
    verify_onnx_model,
)


class TestONNXVerification:
    """End-to-end ONNX verification tests."""
    
    @pytest.fixture
    def actor_setup(self):
        """Create actor and tokenizer for testing."""
        d_model = 32
        config = TransformerConfig(
            d_model=d_model,
            nhead=4,
            num_layers=2,
            dim_feedforward=64,
            action_dim=4
        )
        tok_config = TokenizerConfig(d_model=d_model)
        # Per-agent observation names (2 agents)
        feature_names = [
            ['ev_0_soc', 'ev_0_connected', 'hour', 'price'],
            ['ev_1_soc', 'ev_1_connected', 'hour', 'price'],
        ]
        tokenizer = ObservationTokenizer(tok_config, feature_names)
        actor = TransformerActor(config)
        return actor, tokenizer, config
    
    @pytest.fixture
    def example_tokens(self, actor_setup):
        """Create example token tensors for export."""
        actor, tokenizer, config = actor_setup
        batch_size = 2
        n_ca = 3
        n_sro = 2
        d_model = config.d_model
        
        ca_tokens = torch.randn(batch_size, n_ca, d_model)
        sro_tokens = torch.randn(batch_size, n_sro, d_model)
        nfc_token = torch.randn(batch_size, 1, d_model)
        
        return ca_tokens, sro_tokens, nfc_token
    
    def test_onnx_export_creates_file(self, actor_setup, example_tokens):
        """Test that ONNX export creates a valid file."""
        actor, tokenizer, config = actor_setup
        ca_tokens, sro_tokens, nfc_token = example_tokens
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, 'actor.onnx')
            export_transformer_actor_to_onnx(
                actor=actor,
                output_path=onnx_path,
                example_ca_tokens=ca_tokens,
                example_sro_tokens=sro_tokens,
                example_nfc_token=nfc_token,
            )
            
            assert os.path.exists(onnx_path), "ONNX file not created"
            assert os.path.getsize(onnx_path) > 0, "ONNX file is empty"
    
    def test_onnx_inference_with_onnxruntime(self, actor_setup, example_tokens):
        """Test ONNX inference with onnxruntime."""
        actor, tokenizer, config = actor_setup
        ca_tokens, sro_tokens, nfc_token = example_tokens
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, 'actor.onnx')
            export_transformer_actor_to_onnx(
                actor=actor,
                output_path=onnx_path,
                example_ca_tokens=ca_tokens,
                example_sro_tokens=sro_tokens,
                example_nfc_token=nfc_token,
            )
            
            # Import onnxruntime
            ort = pytest.importorskip("onnxruntime")
            
            # Create session
            session = ort.InferenceSession(onnx_path)
            
            # Get input/output names
            input_names = [i.name for i in session.get_inputs()]
            output_names = [o.name for o in session.get_outputs()]
            
            assert 'ca_tokens' in input_names
            assert 'sro_tokens' in input_names
            assert 'nfc_token' in input_names
            assert 'actions' in output_names
            
            # Run inference
            outputs = session.run(
                None,
                {
                    'ca_tokens': ca_tokens.numpy(),
                    'sro_tokens': sro_tokens.numpy(),
                    'nfc_token': nfc_token.numpy(),
                }
            )
            
            batch_size, n_ca = ca_tokens.shape[:2]
            assert outputs[0].shape == (batch_size, n_ca, config.action_dim)
    
    def test_onnx_pytorch_output_match(self, actor_setup, example_tokens):
        """Test that ONNX and PyTorch produce matching outputs."""
        actor, tokenizer, config = actor_setup
        ca_tokens, sro_tokens, nfc_token = example_tokens
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, 'actor.onnx')
            export_transformer_actor_to_onnx(
                actor=actor,
                output_path=onnx_path,
                example_ca_tokens=ca_tokens,
                example_sro_tokens=sro_tokens,
                example_nfc_token=nfc_token,
            )
            
            # Import onnxruntime
            ort = pytest.importorskip("onnxruntime")
            session = ort.InferenceSession(onnx_path)
            
            # ONNX inference
            onnx_outputs = session.run(
                None,
                {
                    'ca_tokens': ca_tokens.numpy(),
                    'sro_tokens': sro_tokens.numpy(),
                    'nfc_token': nfc_token.numpy(),
                }
            )
            
            # PyTorch inference
            actor.eval()
            with torch.no_grad():
                torch_outputs = actor(ca_tokens, sro_tokens, nfc_token)
            
            # Compare
            max_diff = np.abs(onnx_outputs[0] - torch_outputs.numpy()).max()
            assert max_diff < 1e-4, f"ONNX and PyTorch outputs differ by {max_diff}"
    
    def test_onnx_dynamic_batch_size(self, actor_setup, example_tokens):
        """Test ONNX model with different batch sizes."""
        actor, tokenizer, config = actor_setup
        ca_tokens, sro_tokens, nfc_token = example_tokens
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, 'actor.onnx')
            export_transformer_actor_to_onnx(
                actor=actor,
                output_path=onnx_path,
                example_ca_tokens=ca_tokens,
                example_sro_tokens=sro_tokens,
                example_nfc_token=nfc_token,
            )
            
            ort = pytest.importorskip("onnxruntime")
            session = ort.InferenceSession(onnx_path)
            
            d_model = config.d_model
            n_ca = 3
            n_sro = 2
            
            # Test various batch sizes
            for batch_size in [1, 2, 4]:
                test_ca = np.random.randn(batch_size, n_ca, d_model).astype(np.float32)
                test_sro = np.random.randn(batch_size, n_sro, d_model).astype(np.float32)
                test_nfc = np.random.randn(batch_size, 1, d_model).astype(np.float32)
                
                outputs = session.run(
                    None,
                    {
                        'ca_tokens': test_ca,
                        'sro_tokens': test_sro,
                        'nfc_token': test_nfc,
                    }
                )
                
                assert outputs[0].shape == (batch_size, n_ca, config.action_dim)
    
    def test_onnx_dynamic_sequence_length(self, actor_setup, example_tokens):
        """Test ONNX model with different CA sequence lengths."""
        actor, tokenizer, config = actor_setup
        ca_tokens, sro_tokens, nfc_token = example_tokens
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, 'actor.onnx')
            export_transformer_actor_to_onnx(
                actor=actor,
                output_path=onnx_path,
                example_ca_tokens=ca_tokens,
                example_sro_tokens=sro_tokens,
                example_nfc_token=nfc_token,
            )
            
            ort = pytest.importorskip("onnxruntime")
            session = ort.InferenceSession(onnx_path)
            
            d_model = config.d_model
            batch_size = 2
            n_sro = 2
            
            # Test various CA sequence lengths
            for n_ca in [1, 3, 5, 8]:
                test_ca = np.random.randn(batch_size, n_ca, d_model).astype(np.float32)
                test_sro = np.random.randn(batch_size, n_sro, d_model).astype(np.float32)
                test_nfc = np.random.randn(batch_size, 1, d_model).astype(np.float32)
                
                outputs = session.run(
                    None,
                    {
                        'ca_tokens': test_ca,
                        'sro_tokens': test_sro,
                        'nfc_token': test_nfc,
                    }
                )
                
                assert outputs[0].shape == (batch_size, n_ca, config.action_dim)
    
    def test_onnx_outputs_bounded(self, actor_setup, example_tokens):
        """Test that ONNX outputs are bounded in [-1, 1] due to tanh."""
        actor, tokenizer, config = actor_setup
        ca_tokens, sro_tokens, nfc_token = example_tokens
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, 'actor.onnx')
            export_transformer_actor_to_onnx(
                actor=actor,
                output_path=onnx_path,
                example_ca_tokens=ca_tokens,
                example_sro_tokens=sro_tokens,
                example_nfc_token=nfc_token,
            )
            
            ort = pytest.importorskip("onnxruntime")
            session = ort.InferenceSession(onnx_path)
            
            outputs = session.run(
                None,
                {
                    'ca_tokens': ca_tokens.numpy(),
                    'sro_tokens': sro_tokens.numpy(),
                    'nfc_token': nfc_token.numpy(),
                }
            )
            
            # All outputs should be in valid range [-1, 1] due to tanh
            assert np.all(outputs[0] >= -1.0)
            assert np.all(outputs[0] <= 1.0)
    
    def test_verify_onnx_model(self, actor_setup, example_tokens):
        """Test the verify_onnx_model utility function."""
        pytest.importorskip("onnxruntime")  # Skip if onnxruntime not available
        
        actor, tokenizer, config = actor_setup
        ca_tokens, sro_tokens, nfc_token = example_tokens
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, 'actor.onnx')
            export_transformer_actor_to_onnx(
                actor=actor,
                output_path=onnx_path,
                example_ca_tokens=ca_tokens,
                example_sro_tokens=sro_tokens,
                example_nfc_token=nfc_token,
            )
            
            # Test inputs with different shapes
            test_inputs = [
                (ca_tokens, sro_tokens, nfc_token),
                (torch.randn(1, 2, config.d_model), 
                 torch.randn(1, 1, config.d_model),
                 torch.randn(1, 1, config.d_model)),
            ]
            
            result = verify_onnx_model(onnx_path, test_inputs)
            assert result is True
