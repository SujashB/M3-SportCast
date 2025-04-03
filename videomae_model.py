import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2
import os
from typing import Optional, Tuple, List, Dict

class PatchEmbed3D(nn.Module):
    """
    Video to Patch Embedding for 3D inputs
    """
    def __init__(
        self,
        img_size: int = 224,
        temporal_size: int = 16,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.temporal_size = temporal_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        
        self.grid_size = img_size // patch_size
        self.temporal_grid_size = temporal_size // temporal_patch_size
        self.num_patches = self.grid_size * self.grid_size * self.temporal_grid_size
        
        self.proj = nn.Conv3d(
            in_channels, 
            embed_dim, 
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size)
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        assert H == self.img_size, f"Input height ({H}) doesn't match model ({self.img_size})"
        assert W == self.img_size, f"Input width ({W}) doesn't match model ({self.img_size})"
        assert T == self.temporal_size, f"Input temporal size ({T}) doesn't match model ({self.temporal_size})"
        
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # B, L, C
        x = self.norm(x)
        return x

class Attention(nn.Module):
    """Multi-headed attention implementation"""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, num_heads, N, head_dim

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FeedForward(nn.Module):
    """MLP block in transformer"""
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder for 3D Video Understanding
    Inspired by VideoMAE but simplified for our implementation
    """
    def __init__(
        self,
        img_size: int = 224,
        temporal_size: int = 16,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 528,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
        mask_ratio: float = 0.75,
        num_classes: int = 4,  # 'neutral', 'attack', 'defense', 'lunge'
    ):
        super().__init__()
        
        # Encoder specifics
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            temporal_size=temporal_size,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )
        
        # Add position embeddings
        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=True, 
                norm_layer=norm_layer
            )
            for _ in range(depth)
        ])
        
        # Encoder to decoder projection
        self.norm = norm_layer(embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
        # Decoder position embeddings
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim))
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer
            )
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        # We have dual heads:
        # 1. Reconstruction head for self-supervised learning
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, 
            temporal_patch_size * patch_size * patch_size * in_channels, 
            bias=True
        )
        
        # 2. Classification head for fencing techniques
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Parameters
        self.mask_ratio = mask_ratio
        self.img_size = img_size
        self.temporal_size = temporal_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        
        # Initialize weights
        self.initialize_weights()
        
        # Class names
        self.class_names = ['neutral', 'attack', 'defense', 'lunge']
        
    def initialize_weights(self):
        # Initialize position embeddings
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            self.patch_embed.grid_size,  # Use correct spatial grid size
            t_size=self.patch_embed.temporal_grid_size,  # Use correct temporal grid size
            cls_token=False
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        decoder_pos_embed = get_3d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], 
            self.patch_embed.grid_size,  # Use correct spatial grid size
            t_size=self.patch_embed.temporal_grid_size, # Use correct temporal grid size
            cls_token=False
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        # Initialize cls token
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize all other weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
            
    def random_masking(
        self, 
        x: torch.Tensor, 
        mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform random masking by per-sample shuffling.
        
        Args:
            x: Input tokens, shape (B, L, D)
            mask_ratio: Masking ratio (0 to 1)
            
        Returns:
            masked tokens, mask, ids_restore
        """
        B, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        # Generate noise for random shuffling
        noise = torch.rand(B, L, device=x.device)
        
        # Sort indices based on noise
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first len_keep elements
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate the binary mask: 0 for keep, 1 for remove
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(
        self, 
        x: torch.Tensor, 
        mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder with masking
        """
        # Tokenize
        x = self.patch_embed(x)
        B, L, C = x.shape
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Apply masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Append cls token for classification task
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        # Final normalization
        x = self.norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(
        self, 
        x: torch.Tensor, 
        ids_restore: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the decoder
        """
        # Skip cls token for reconstruction
        x = x[:, 1:, :]
        
        # Project from encoder to decoder dimensions
        x = self.decoder_embed(x)
        B, L, C = x.shape
        
        # Unshuffle tokens to original positions
        mask_tokens = torch.zeros(
            [B, self.num_patches - L, C], device=x.device
        )
        x_ = torch.cat([x, mask_tokens], dim=1)  # Append mask tokens
        x = torch.gather(
            x_, dim=1, 
            index=ids_restore.unsqueeze(-1).repeat(1, 1, C)
        )  # Unshuffle
        
        # Add position embeddings
        x = x + self.decoder_pos_embed
        
        # Apply decoder transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        
        # Final normalization
        x = self.decoder_norm(x)
        
        # Predict pixel values
        x = self.decoder_pred(x)
        
        return x
    
    def forward_classifier(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classification head
        """
        # Use [CLS] token for classification
        x = x[:, 0]
        x = self.cls_head(x)
        return x
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask_ratio: float = None
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass with both reconstruction and classification
        """
        # Use provided mask ratio or default
        mask_ratio = mask_ratio if mask_ratio is not None else self.mask_ratio
        
        # Get encoded features with masking
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        
        # Get predicted patches from decoder
        pred = self.forward_decoder(latent, ids_restore)
        
        # Get classification output
        cls_output = self.forward_classifier(latent)
        
        return {
            'latent': latent[:, 0],  # [CLS] token features
            'pred': pred,
            'mask': mask,
            'cls_output': cls_output
        }
    
    def predict(self, x: torch.Tensor) -> Tuple[int, str, float]:
        """
        Make a prediction on a single video tensor
        
        Args:
            x: Preprocessed video tensor of shape [1, C, T, H, W]
            
        Returns:
            class_id: Predicted class ID
            class_name: Name of predicted class
            confidence: Confidence score
        """
        self.eval()
        with torch.no_grad():
            # Forward pass with no masking (for inference)
            outputs = self.forward(x, mask_ratio=0.0)
            cls_output = outputs['cls_output']
            
            # Get probabilities and prediction
            probabilities = F.softmax(cls_output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            return predicted.item(), self.class_names[predicted.item()], confidence.item()
            
    @torch.no_grad()        
    def get_temporal_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract temporal attention weights for visualization
        """
        self.eval()
        
        # Tokenize
        x = self.patch_embed(x)
        B, L, C = x.shape
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # No masking for attention visualization
        
        # Append cls token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Get attention weights from the last block
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # For the last block, manually execute to get attention weights
                x_norm = blk.norm1(x)
                
                # Get Q, K, V tensors
                B, N, C = x_norm.shape
                qkv = blk.attn.qkv(x_norm).reshape(B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                
                # Calculate attention scores
                attn = (q @ k.transpose(-2, -1)) * blk.attn.scale
                attn = attn.softmax(dim=-1)  # B, num_heads, N, N
                
                # Return attention from the CLS token to all other tokens
                # First row of attention matrix for each head
                cls_attn = attn[:, :, 0, 1:]  # B, num_heads, L
                
                return cls_attn


class FencingTemporalModel(nn.Module):
    """
    Wrapper class for the 3D video model specifically for fencing technique classification
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        temporal_size: int = 16,
        img_size: int = 224,
        num_classes: int = 4,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.temporal_size = temporal_size
        self.img_size = img_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the core model (e.g., VideoMAE)
        self.model = MaskedAutoencoder(
            img_size=self.img_size,
            temporal_size=self.temporal_size,
            num_classes=num_classes
        ).to(self.device)
        
        # Load pretrained weights if path is provided
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                # Handle potential state dict mismatches (e.g., from different training phases)
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                # Filter out unnecessary keys if needed
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
                # Load the state dict
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    print(f"Warning: Missing keys in state dict: {missing_keys}")
                if unexpected_keys:
                    print(f"Warning: Unexpected keys in state dict: {unexpected_keys}")
                print(f"Loaded temporal model weights from: {model_path}")
            except Exception as e:
                print(f"Error loading model weights from {model_path}: {e}")
                print("Initializing with random weights instead.")
        else:
            print("Initializing fencing temporal model with random weights")
            
        # Store class names if available (passed from dataset)
        # Note: The model itself inside MaskedAutoencoder has a default list
        # This allows overriding if the dataset has different classes
        self.class_names = [self.model.class_names[i] if i < len(self.model.class_names) else f'class_{i}' for i in range(num_classes)]
        
        # For data preprocessing
        self.mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1, 1).to(self.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for the FencingTemporalModel.
        For supervised training/inference, we only need the classifier output.
        """
        # Pass input through the encoder and get the classification output
        # We set mask_ratio=0.0 for inference/supervised training as we don't mask inputs
        latent, _, _ = self.model.forward_encoder(x, mask_ratio=0.0)
        cls_output = self.model.forward_classifier(latent)
        return cls_output

    def preprocess_video(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Preprocess a list of frames into a tensor suitable for the model"""
        # Check if we have enough frames
        if len(frames) < self.temporal_size:
            # If not enough frames, repeat the last frame
            frames = frames + [frames[-1]] * (self.temporal_size - len(frames))
        elif len(frames) > self.temporal_size:
            # If too many frames, sample evenly
            indices = np.linspace(0, len(frames) - 1, self.temporal_size, dtype=int)
            frames = [frames[i] for i in indices]
        
        # Convert BGR to RGB and resize
        processed_frames = []
        for frame in frames:
            # Convert BGR to RGB
            frame_rgb = frame[..., ::-1]  # BGR to RGB
            
            # Resize frame
            frame_resized = cv2.resize(frame_rgb, (self.img_size, self.img_size))
            
            # Convert to [0, 1] range float
            frame_float = frame_resized.astype(np.float32) / 255.0
            
            processed_frames.append(frame_float)
        
        # Stack frames to form a video array [T, H, W, C]
        video_array = np.stack(processed_frames, axis=0)
        
        # Convert to tensor and permute to [C, T, H, W]
        video_tensor = torch.from_numpy(video_array).permute(3, 0, 1, 2).float().to(self.device)
        
        # Add batch dimension [1, C, T, H, W]
        video_tensor = video_tensor.unsqueeze(0)
        
        # Normalize
        video_tensor = (video_tensor - self.mean) / self.std
        
        return video_tensor
    
    def classify_sequence(
        self, 
        frames: List[np.ndarray]
    ) -> Tuple[int, str, float]:
        """
        Classify a sequence of frames
        
        Args:
            frames: List of video frames (each frame is a NumPy array in BGR format)
            
        Returns:
            class_id: ID of predicted class
            class_name: Name of predicted class
            confidence: Confidence score
        """
        # Preprocess frames
        video_tensor = self.preprocess_video(frames)
        
        # Make prediction
        class_id, class_name, confidence = self.model.predict(video_tensor)
        
        return class_id, class_name, confidence
    
    def get_temporal_attention(self, frames: List[np.ndarray]) -> np.ndarray:
        """Get temporal attention weights for visualization"""
        self.model.eval()
        video_tensor = self.preprocess_video(frames)
        
        with torch.no_grad():
            # Get encoder output
            latent, _, _ = self.model.forward_encoder(video_tensor, mask_ratio=0.0)
            
            # Get attention from the last block (can modify to average or select)
            last_block_attention = self.model.blocks[-1].attn
            
            # Calculate attention weights (simplified example)
            # This needs refinement based on how attention is implemented
            B, N, C = latent.shape
            # Use the already normalized latent output from the encoder
            qkv = last_block_attention.qkv(latent).reshape(
                B, N, 3, last_block_attention.num_heads, C // last_block_attention.num_heads
            ).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn_matrix = (q @ k.transpose(-2, -1)) * last_block_attention.scale
            attn_matrix = attn_matrix.softmax(dim=-1)
            
            # Average attention across heads and maybe tokens (needs careful consideration)
            temporal_attention = attn_matrix.mean(dim=1)[:, 0, 1:].cpu().numpy() # Example: attention to CLS token
            
        return temporal_attention

    def train_model_on_data(self, train_loader, val_loader, epochs=10, lr=1e-4, output_dir='models'):
        """
        Train the temporal model
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs to train
            lr: Learning rate
            output_dir: Directory to save best model
        """
        # Set model to training mode
        self.model.train()
        
        # Define loss functions and optimizer
        recon_criterion = nn.MSELoss()
        cls_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=0.05
        )
        
        # Learning rate schedule
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        
        best_val_acc = 0.0
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_recon_loss = 0.0
            train_cls_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for videos, labels in train_loader:
                videos, labels = videos.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(videos, mask_ratio=0.75)
                
                # Reconstruction loss
                recon_loss = recon_criterion(
                    outputs['pred'], 
                    videos.view(videos.shape[0], -1, videos.shape[2] * videos.shape[3] * videos.shape[4])
                )
                
                # Classification loss
                cls_loss = cls_criterion(outputs['cls_output'], labels)
                
                # Combined loss with weighting
                loss = 0.7 * recon_loss + 0.3 * cls_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_recon_loss += recon_loss.item() * videos.size(0)
                train_cls_loss += cls_loss.item() * videos.size(0)
                _, predicted = torch.max(outputs['cls_output'], 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_recon_loss = train_recon_loss / train_total
            train_cls_loss = train_cls_loss / train_total
            train_acc = train_correct / train_total
            
            # Update learning rate
            lr_scheduler.step()
            
            # Validation phase
            if val_loader:
                self.model.eval()
                val_recon_loss = 0.0
                val_cls_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for videos, labels in val_loader:
                        videos, labels = videos.to(self.device), labels.to(self.device)
                        
                        # Forward pass
                        outputs = self.model(videos, mask_ratio=0.0)  # No masking for validation
                        
                        # Reconstruction loss
                        recon_loss = recon_criterion(
                            outputs['pred'], 
                            videos.view(videos.shape[0], -1, videos.shape[2] * videos.shape[3] * videos.shape[4])
                        )
                        
                        # Classification loss
                        cls_loss = cls_criterion(outputs['cls_output'], labels)
                        
                        # Statistics
                        val_recon_loss += recon_loss.item() * videos.size(0)
                        val_cls_loss += cls_loss.item() * videos.size(0)
                        _, predicted = torch.max(outputs['cls_output'], 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                val_recon_loss = val_recon_loss / val_total
                val_cls_loss = val_cls_loss / val_total
                val_acc = val_correct / val_total
                
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Recon Loss: {train_recon_loss:.4f}, Train Cls Loss: {train_cls_loss:.4f}, "
                      f"Train Acc: {train_acc:.4f}, "
                      f"Val Recon Loss: {val_recon_loss:.4f}, Val Cls Loss: {val_cls_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}")
                
                # Save best model
                if val_acc > best_val_acc and output_dir:
                    best_val_acc = val_acc
                    torch.save(self.model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                    print(f"Saved model with validation accuracy: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Recon Loss: {train_recon_loss:.4f}, Train Cls Loss: {train_cls_loss:.4f}, "
                      f"Train Acc: {train_acc:.4f}")
                
                # Save the model periodically
                if (epoch + 1) % 5 == 0 and output_dir:
                    save_path_epoch = os.path.join(output_dir, f"epoch{epoch+1}_model.pth")
                    torch.save(self.model.state_dict(), save_path_epoch)
                    print(f"Saved model checkpoint at epoch {epoch+1}")
        
        # Load best model if saved
        if output_dir and os.path.exists(os.path.join(output_dir, 'best_model.pth')):
            self.model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth'), map_location=self.device))
            print(f"Loaded best model with validation accuracy: {best_val_acc:.4f}")
        
        return self.model


def get_3d_sincos_pos_embed(embed_dim, grid_size, t_size, cls_token=False):
    """
    3D sine-cosine positional embedding for 3D patched input.
    
    Args:
        embed_dim: Output dimension for each position
        grid_size: The spatial grid size (H/P)
        t_size: The temporal grid size (T/P_t)
        cls_token: Whether to include cls token position embedding
        
    Returns:
        pos_embed: (L, D) positional embeddings
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid_t = np.arange(t_size, dtype=np.float32)
    
    # Calculate grid coordinates
    grid = np.meshgrid(grid_w, grid_h, grid_t, indexing='ij')  # w, h, t
    grid = np.stack(grid, axis=0)
    
    # Normalize grid coordinates
    grid = grid.reshape([3, 1, t_size, grid_size, grid_size]) # Match VideoMAE implementation
    
    # Compute embeddings
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    # Add cls token if needed
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
        
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Compute 3D positional embeddings from grid coordinates"""
    assert embed_dim % 6 == 0, 'Embed dimension must be divisible by 6 for 3D sin-cos'
    
    # Flatten the grid dimensions (W, H, T)
    t_size, grid_h, grid_w = grid.shape[-3:]
    grid_w_coords = grid[0].reshape(-1) # Flatten W coordinates
    grid_h_coords = grid[1].reshape(-1) # Flatten H coordinates
    grid_t_coords = grid[2].reshape(-1) # Flatten T coordinates
    
    # Use 1/3 of dimensions for each axis
    emb_dim_per_axis = embed_dim // 3
    emb_w = get_1d_sincos_pos_embed_from_grid(emb_dim_per_axis, grid_w_coords)
    emb_h = get_1d_sincos_pos_embed_from_grid(emb_dim_per_axis, grid_h_coords)
    emb_t = get_1d_sincos_pos_embed_from_grid(emb_dim_per_axis, grid_t_coords)
    
    # Concatenate embeddings
    pos_embed = np.concatenate([emb_w, emb_h, emb_t], axis=1)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    1D sine-cosine positional embedding
    
    Args:
        embed_dim: Output dimension for each position
        pos: Position indices, shape (L,)
        
    Returns:
        pos_embed: (L, D) positional embeddings
    """
    assert embed_dim % 2 == 0, "Embed dimension must be even"
    
    # Get sequence length
    L = pos.shape[0]
    
    # Get omega values
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)  # (D/2)

    # Compute positional embeddings
    pos = pos.reshape(-1)  # (L)
    out = np.einsum('i,j->ij', pos, omega)  # (L, D/2)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    pos_embed = np.concatenate([emb_sin, emb_cos], axis=1)
    return pos_embed


def create_video_dataloader(
    video_dir: str,
    batch_size: int = 4,
    temporal_size: int = 16,
    img_size: int = 224,
    train_ratio: float = 0.8,
    num_workers: int = 4
):
    """
    Create DataLoader for training and validation from video datasets
    
    Args:
        video_dir: Path to directory containing class subdirectories of videos
        batch_size: Batch size
        temporal_size: Number of frames to sample per video
        img_size: Image size for training
        train_ratio: Ratio of data to use for training
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
    """
    # TODO: Implement custom video data loading from directory
    # For now, this is a placeholder
    pass 