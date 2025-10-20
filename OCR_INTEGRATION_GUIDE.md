# OCR Loss Integration Guide

## Structure

```
models/
  my_ocr/
    __init__.py           # Empty init file
    ocr_module.py         # Contains SimpleOCR and OCRLoss classes
```

## What Was Changed

### 1. Created OCR Module (`models/my_ocr/ocr_module.py`)
- **SimpleOCR**: Uses EasyOCR to extract text from images
- **OCRLoss**: Computes text preservation loss using character-level Jaccard similarity

### 2. Updated Training Pipeline (`train.py`)
- Import: `from models.my_ocr.ocr_module import OCRLoss`
- Initialize OCR loss with configurable weight
- Compute OCR loss with gradients enabled for `fake_B`
- Backpropagate OCR loss through generator
- Add OCR loss to logging

## Key Implementation Details

### ✅ Gradient Flow
```python
# fake_B keeps gradients, real_A is detached
ocr_loss = ocr_criterion(fake_B, real_A.detach())
ocr_loss.backward(retain_graph=True)
model.optimizer_G.step()
```

### ✅ Loss Logging
```python
# Add to model's loss names
if 'OCR' not in model.loss_names:
    model.loss_names.append('OCR')

# Store for logging
model.loss_OCR = ocr_loss.item()

# Include in visualization
losses = model.get_current_losses()
if hasattr(model, 'loss_OCR'):
    losses['OCR'] = model.loss_OCR
```

## Installation

1. Install EasyOCR:
```bash
pip install easyocr
```

2. The OCR model will automatically download language packs on first use.

## Usage

### Enable OCR Loss (default weight 0.1)
```bash
python train.py --dataroot ./datasets/your_dataset --name experiment_name
```

### Custom OCR Loss Weight
```bash
python train.py --dataroot ./datasets/your_dataset --name experiment_name --lambda_OCR 0.5
```

### Disable OCR Loss
```bash
python train.py --dataroot ./datasets/your_dataset --name experiment_name --lambda_OCR 0
```

## How It Works

1. **Text Extraction**: EasyOCR extracts text from both real (source) and fake (generated) images
2. **Similarity Computation**: Character-level Jaccard similarity measures text preservation
3. **Loss Calculation**: Loss = 1 - similarity (minimize text dissimilarity)
4. **Backpropagation**: Gradients flow through generator to preserve text during style transfer

## Training Output

You'll see OCR loss in the training logs:
```
(epoch: 1, iters: 100) G_GAN: 0.856 D_real: 0.432 D_fake: 0.123 G: 1.234 NCE: 0.567 OCR: 0.234
```

## Notes

- OCR loss is computed **before** `optimize_parameters()` to ensure proper gradient flow
- Uses `retain_graph=True` to allow multiple backward passes
- Real images are detached to prevent unnecessary gradient computation
- OCR text extraction is wrapped in `torch.no_grad()` for efficiency
