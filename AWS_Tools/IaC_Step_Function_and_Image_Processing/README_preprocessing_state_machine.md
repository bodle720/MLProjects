## Image Preprocessing State Machine Overview

This state machine orchestrates a modular image preprocessing pipeline using one Zip and four Docker-based AWS Lambda functions. Each function is invoked conditionally via a Choice state, allowing flexible, audit-friendly transformations in the following order:

ResizeLambda → ContrastEnhanceCLAHELambda → MakeGrayscaleLambda → NormalizeLambda

---

### 1️. ResizeLambda

Resizes an input image from S3 using configurable parameters.  
Supports aspect-ratio-preserving scaling or forced resizing.

**Parameters:**
- method:
  - 'preserve_aspect_ratio': scales both dimensions by scale_factor
  - 'force': resizes to new_width × new_height (ignores aspect ratio)
- scale_factor: string float (e.g. "1.20" for +20%, "0.80" for −20%)
- new_width, new_height: integers used when method='force'
- resize_order: integer interpolation method (0–5)
  - 0: Nearest-neighbor
  - 1: Bi-linear
  - 2: Bi-quadratic
  - 3: Bi-cubic
  - 4: Bi-quartic
  - 5: Bi-quintic

**Notes:**
- Always uses preserve_range=False to scale output to [0.0, 1.0] for downstream CLAHE compatibility.

---

### 2️. ContrastEnhanceCLAHELambda

Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance local contrast.

**Parameters:**
- kernel_size: integer (e.g. 8, 16, 32, 64) → interpreted as (k, k)
  - (8, 8) or (16, 16): small/medium grayscale images
  - (32, 32) or (64, 64): high-res images or smoother transitions
  - Smaller kernels: more aggressive enhancement, possible noise
  - Larger kernels: smoother output, less local detail
- clip_limit: string float (e.g. "0.05")


**Reference Table:**

| clip_limit | Effect              | Use Case                                                  |
|------------|---------------------|------------------------------------------------------------|
| 0.01–0.03  | Mild enhancement    | Natural images, medical scans, low-noise inputs           |
| 0.03–0.1   | Moderate contrast   | General-purpose enhancement, grayscale textures           |
| >0.1       | Aggressive contrast | High-detail images, but may introduce artifacts or noise  |

---
### 3️. MakeGrayscaleLambda

Converts RGB images to single-band grayscale using luminance weighting.

**Behavior:**
- Applies rgb2gray from scikit-image if input shape is (H, W, 3). Otherwise, does nothing.
- Output shape: (H, W)
- Formula: Gray = 0.2125 * R + 0.7154 * G + 0.0721 * B

---

### 4️. NormalizeLambda

Normalizes pixel values to either [0.0, 1.0] (float) or [0, 255] (uint8).

**Parameters:**
- normalization:
  - '0to1': float normalization
  - '0to255': uint8 normalization

---
