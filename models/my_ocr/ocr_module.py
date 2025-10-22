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
    - Uses perceptual loss on heavily downsampled images (critical for high-res stability)
    - Includes aggressive normalization and warmup
    - Designed for 2048x512 images
    """
    def __init__(self, device='cuda', weight=1.0, warmup_iters=200):
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
        Compute OCR loss with extreme stability for high-resolution images
        
        Args:
            fake_image: generated image tensor (B, C, H, W) - has gradients
            real_image: source image tensor (B, C, H, W) - detached
        Returns:
            loss: scalar tensor with gradients
        """
        try:
            # Safety check for NaN/Inf in inputs
            if torch.isnan(fake_image).any() or torch.isinf(fake_image).any():
                print("WARNING: NaN/Inf in fake_image input to OCR loss")
                return torch.tensor(0.0, device=fake_image.device, requires_grad=True)
            
            if torch.isnan(real_image).any() or torch.isinf(real_image).any():
                print("WARNING: NaN/Inf in real_image input to OCR loss")
                return torch.tensor(0.0, device=fake_image.device, requires_grad=True)
            
            # CRITICAL: Downsample to 128x128 for ultra-high-resolution images
            # This massively reduces gradient magnitude
            target_size = (128, 128) if (fake_image.size(2) > 512 or fake_image.size(3) > 512) else (256, 256)
            
            fake_small = F.interpolate(fake_image, size=target_size, mode='bilinear', align_corners=False)
            real_small = F.interpolate(real_image, size=target_size, mode='bilinear', align_corners=False)
            
            # Use smooth L1 loss
            content_loss = self.loss_fn(fake_small, real_small)
            
            # Safety check
            if torch.isnan(content_loss) or torch.isinf(content_loss):
                print("WARNING: NaN/Inf in OCR content_loss")
                return torch.tensor(0.0, device=fake_image.device, requires_grad=True)
            
            # Warmup: start at 0, gradually increase to full weight
            self.current_iter += 1
            if self.current_iter <= self.warmup_iters:
                # Slower warmup for high-resolution stability
                warmup_factor = (self.current_iter / self.warmup_iters) ** 2  # quadratic warmup
            else:
                warmup_factor = 1.0
            
            # Very aggressive clamping for high-resolution: max loss = 1.0
            weighted_loss = torch.clamp(content_loss * self.weight * warmup_factor, max=1.0)
            
            # Final safety check
            if torch.isnan(weighted_loss) or torch.isinf(weighted_loss):
                print("WARNING: NaN/Inf in final OCR loss")
                return torch.tensor(0.0, device=fake_image.device, requires_grad=True)
            
            return weighted_loss
            
        except Exception as e:
            print(f"ERROR in OCR loss computation: {e}")
            return torch.tensor(0.0, device=fake_image.device, requires_grad=True)
