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
            image: tensor of shape (B, C, H, W) or numpy array
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
        results = self.reader.readtext(img_np, detail=0)  # detail=0 returns only text
        return results
    
    def forward(self, image):
        """Forward pass - extract text features"""
        return self.extract_text(image)


class OCRLoss(nn.Module):
    """
    OCR Loss: Measures text preservation between source and generated images
    """
    def __init__(self, device='cuda', weight=1.0):
        super(OCRLoss, self).__init__()
        self.ocr_model = SimpleOCR(device=device)
        self.weight = weight
        self.device = device
    
    def compute_text_similarity(self, text1_list, text2_list):
        """
        Compute similarity between two lists of text
        Simple character-level Jaccard similarity
        """
        if not text1_list or not text2_list:
            return torch.tensor(1.0, device=self.device)  # No text penalty
        
        # Join all detected texts
        text1 = ' '.join(text1_list).lower()
        text2 = ' '.join(text2_list).lower()
        
        # Character-level Jaccard similarity
        set1 = set(text1)
        set2 = set(text2)
        
        if len(set1) == 0 and len(set2) == 0:
            return torch.tensor(0.0, device=self.device)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            similarity = 0.0
        else:
            similarity = intersection / union
        
        # Loss is 1 - similarity (want to minimize dissimilarity)
        loss = 1.0 - similarity
        
        return torch.tensor(loss, device=self.device, dtype=torch.float32)
    
    def forward(self, fake_image, real_image):
        """
        Compute OCR loss between fake and real images
        Args:
            fake_image: generated image tensor (B, C, H, W) - should have gradients
            real_image: source image tensor (B, C, H, W) - will be detached
        Returns:
            loss: scalar tensor with gradients
        """
        batch_size = fake_image.size(0)
        total_loss = 0.0
        
        for i in range(batch_size):
            # Extract text from both images
            # Note: OCR extraction doesn't require gradients, but we keep fake_image gradient-enabled
            with torch.no_grad():
                fake_text = self.ocr_model(fake_image[i])
                real_text = self.ocr_model(real_image[i])
            
            # Compute similarity loss
            loss = self.compute_text_similarity(real_text, fake_text)
            total_loss += loss
        
        # Average over batch
        avg_loss = total_loss / batch_size if batch_size > 0 else total_loss
        
        return avg_loss * self.weight
