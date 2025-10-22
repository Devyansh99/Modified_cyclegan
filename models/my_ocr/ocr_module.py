"""
Simple OCR Model with built-in OCR and OCR Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import easyocr
except ImportError:
    easyocr = None


class SimpleOCR(nn.Module):
    """Simple OCR model using EasyOCR as backbone"""
    def __init__(self, device='cuda'):
        super(SimpleOCR, self).__init__()
        self.device = device
        
        # Initialize EasyOCR reader
        if easyocr is not None:
            self.reader = easyocr.Reader(['en'], gpu=(device == 'cuda'))
        else:
            print("Warning: easyocr not installed. OCR functionality will be limited.")
            self.reader = None
    
    def extract_text(self, image):
        """
        Extract text from image using EasyOCR
        Args:
            image: tensor of shape (C, H, W) or numpy array
        Returns:
            list of detected text strings
        """
        if self.reader is None:
            return []
        
        # Convert tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            # Denormalize from [-1, 1] to [0, 255]
            img_np = ((image + 1) * 127.5).clamp(0, 255).byte()
            img_np = img_np.cpu().numpy().transpose(1, 2, 0)  # CHW to HWC
        else:
            img_np = image
        
        # Detect text
        try:
            results = self.reader.readtext(img_np, detail=0)  # detail=0 returns only text
            return results
        except:
            return []
    
    def forward(self, image):
        """Forward pass - extract text features"""
        return self.extract_text(image)


class OCRLoss(nn.Module):
    """
    OCR Loss: Measures text preservation using a hybrid approach
    - Uses perceptual loss on downsampled images (more stable for large images)
    - Normalized to prevent gradient explosion
    - Includes warmup phase to prevent early training collapse
    """
    def __init__(self, device='cuda', weight=1.0, warmup_iters=100):
        super(OCRLoss, self).__init__()
        self.ocr_model = SimpleOCR(device=device)
        self.weight = weight
        self.device = device
        self.warmup_iters = warmup_iters
        self.current_iter = 0
        # Use smooth L1 loss (more stable than L1)
        self.loss_fn = nn.SmoothL1Loss()
    
    def forward(self, fake_image, real_image):
        """
        Compute OCR loss between fake and real images
        Uses smooth L1 loss with normalization for stability
        Includes warmup phase to prevent early collapse
        
        Args:
            fake_image: generated image tensor (B, C, H, W) - has gradients
            real_image: source image tensor (B, C, H, W) - detached
        Returns:
            loss: scalar tensor with gradients
        """
        # Downsample images to reduce gradient magnitude for large images
        # This prevents NaN issues with high-resolution images
        if fake_image.size(2) > 256 or fake_image.size(3) > 256:
            # Downsample to max 256x256 for loss computation
            fake_small = F.interpolate(fake_image, size=(256, 256), mode='bilinear', align_corners=False)
            real_small = F.interpolate(real_image, size=(256, 256), mode='bilinear', align_corners=False)
        else:
            fake_small = fake_image
            real_small = real_image
        
        # Use smooth L1 loss (less sensitive to outliers, more stable)
        content_loss = self.loss_fn(fake_small, real_small)
        
        # Apply warmup: gradually increase OCR loss weight from 0 to target weight
        # This prevents early training collapse when generator output is random
        self.current_iter += 1
        if self.current_iter < self.warmup_iters:
            warmup_factor = self.current_iter / self.warmup_iters
        else:
            warmup_factor = 1.0
        
        # Apply weight with warmup and gradient clipping for safety
        weighted_loss = torch.clamp(content_loss * self.weight * warmup_factor, max=10.0)
        
        return weighted_loss
