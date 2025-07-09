# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mutual_info_score

class PNNSparsifier:
    def __init__(self, model, compression_ratio=0.8, sensitivity_threshold=0.1):
        """
        Compress a neural network using probabilistic pruning based on mutual information.
        
        Args:
            model: PyTorch neural network model to compress
            compression_ratio: Target compression ratio (0.8 = 80% of connections removed)
            sensitivity_threshold: Minimum mutual information required to keep a connection
        """
        self.model = model
        self.compression_ratio = compression_ratio
        self.sensitivity_threshold = sensitivity_threshold
        self.mask_dict = {}  # Stores pruning masks for each layer
        
    def _estimate_mutual_information(self, weight_tensor, output_tensor):
        """Estimate mutual information between weights and output activations"""
        # Flatten tensors for MI calculation
        weights_flat = weight_tensor.detach().cpu().numpy().flatten()
        output_flat = output_tensor.detach().cpu().numpy().flatten()
        
        # Discretize values for mutual information calculation
        weights_discrete = np.digitize(weights_flat, np.histogram_bin_edges(weights_flat, bins=20))
        output_discrete = np.digitize(output_flat, np.histogram_bin_edges(output_flat, bins=20))
        
        # Calculate mutual information
        mi = mutual_info_score(weights_discrete, output_discrete)
        return mi
    
    def _compute_connection_saliency(self, layer_name, weight_tensor, activation_hook):
        """Compute saliency scores for each connection in a layer"""
        # Register hook to capture output activations
        activations = []
        def hook(module, input, output):
            activations.append(output)
        
        hook_handle = activation_hook.register_forward_hook(hook)
        
        # Pass dummy input to capture activations
        dummy_input = torch.randn(1, *self.model.input_shape)
        self.model(dummy_input)
        
        hook_handle.remove()  # Remove hook
        
        # Calculate mutual information for each weight
        output_tensor = activations[0]
        mi_matrix = torch.zeros_like(weight_tensor)
        
        # This can be optimized with vectorization for larger layers
        for i in range(weight_tensor.shape[0]):
            for j in range(weight_tensor.shape[1]):
                # Estimate MI for this specific connection
                mi_matrix[i, j] = self._estimate_mutual_information(
                    weight_tensor[i, j].unsqueeze(0).unsqueeze(0),
                    output_tensor[:, i]
                )
        
        # Combine MI with weight magnitude for saliency
        saliency = torch.abs(weight_tensor) * mi_matrix
        return saliency
    
    def compress(self):
        """Apply probabilistic pruning to the model"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                weight_tensor = module.weight.data
                layer_name = name
                
                # Compute saliency scores for each connection
                saliency = self._compute_connection_saliency(
                    layer_name, weight_tensor, module
                )
                
                # Determine pruning threshold
                flat_saliency = saliency.flatten()
                threshold = np.percentile(flat_saliency.cpu().numpy(), 
                                         self.compression_ratio * 100)
                
                # Create binary mask
                mask = (saliency >= threshold) | (torch.abs(weight_tensor) > self.sensitivity_threshold)
                self.mask_dict[layer_name] = mask
                
                # Apply mask to weights
                module.weight.data *= mask.float()
                
        return self.model
    
    def apply_mask(self):
        """Apply precomputed masks to the model (for inference)"""
        for name, module in self.model.named_modules():
            if name in self.mask_dict:
                module.weight.data *= self.mask_dict[name].float()
    
    def get_compression_stats(self):
        """Return statistics about the compression"""
        original_params = 0
        pruned_params = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                if name in self.mask_dict:
                    original_params += module.weight.numel()
                    pruned_params += (self.mask_dict[name] == 0).sum().item()
        
        compression_ratio = pruned_params / original_params
        return {
            'original_params': original_params,
            'pruned_params': pruned_params,
            'compression_ratio': compression_ratio,
            'size_reduction': f"{compression_ratio * 100:.2f}%"
        }

# Example usage with a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.input_shape = (3, 32, 32)  # For dummy input generation
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Test the compression
if __name__ == "__main__":
    # Create model
    model = SimpleCNN()
    
    # Initialize compressor
    compressor = PNNSparsifier(model, compression_ratio=0.75, sensitivity_threshold=0.05)
    
    # Compress the model
    compressed_model = compressor.compress()
    
    # Print compression statistics
    stats = compressor.get_compression_stats()
    print(f"Model compressed: {stats['size_reduction']} reduction in parameters")
    
    # Test inference with compressed model
    dummy_input = torch.randn(1, 3, 32, 32)
    output = compressed_model(dummy_input)
    print(f"Output shape: {output.shape}")
