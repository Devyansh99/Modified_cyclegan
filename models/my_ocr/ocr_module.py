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
    - Uses L1 loss on image content (differentiable)
    - Optionally logs OCR text similarity for monitoring
    """
    def __init__(self, device='cuda', weight=1.0):
        super(OCRLoss, self).__init__()
        self.ocr_model = SimpleOCR(device=device)
        self.weight = weight
        self.device = device
        # Use L1 loss for differentiable pixel-level similarity
        self.l1_criterion = nn.L1Loss()
    
    def forward(self, fake_image, real_image):
        """
        Compute OCR loss between fake and real images
        Uses L1 loss as a differentiable proxy for text preservation
        
        Args:
            fake_image: generated image tensor (B, C, H, W) - has gradients
            real_image: source image tensor (B, C, H, W) - detached
        Returns:
            loss: scalar tensor with gradients
        """
        # Use L1 loss on image content as differentiable proxy
        # This encourages the generator to preserve overall structure including text
        content_loss = self.l1_criterion(fake_image, real_image)
        
        # Weight the loss
        weighted_loss = content_loss * self.weight
        
        return weighted_loss
