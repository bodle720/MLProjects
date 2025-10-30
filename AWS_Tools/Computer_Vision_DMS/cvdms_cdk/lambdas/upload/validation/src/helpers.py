import io, hashlib, logging
import numpy as np
from PIL import Image
import imagehash
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern
from skimage.filters import sobel, laplace
from skimage.measure import shannon_entropy

logger = logging.getLogger(__name__)

def load_image(file_bytes):
    """Load image from raw bytes into Pillow Image."""
    try:
        return Image.open(io.BytesIO(file_bytes))
    except Exception as e:
        logger.error(f"Image load failed: {e}")
        return None

def validate_bands(img):
    """Check if image is grayscale (L) or RGB."""
    return img.mode in ("L", "RGB")

def compute_sha256(file_bytes):
    """Compute SHA256 hash of raw file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()

def compute_phashes(img):
    """Compute perceptual hashes per band."""
    if img.mode == "L":
        return str(imagehash.phash(img))
    elif img.mode == "RGB":
        bands = img.split()
        return "_".join(str(imagehash.phash(b)) for b in bands)
    else:
        return None

def extract_features(img):
    """Compute statistical and texture features."""
    arr = np.array(img)

    # Basic stats
    stats = {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std())
    }

    # Entropy
    stats["entropy"] = float(shannon_entropy(arr))

    # Edge density (Sobel)
    edges = sobel(arr if arr.ndim == 2 else arr[...,0])
    stats["edge_density"] = float((edges > 0.1).mean())

    # Sharpness (variance of Laplacian)
    lap = laplace(arr if arr.ndim == 2 else arr[...,0])
    stats["laplacian_var"] = float(lap.var())

    # GLCM features (on grayscale)
    gray = arr if arr.ndim == 2 else np.mean(arr, axis=2).astype(np.uint8)
    glcm = greycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    stats["glcm_contrast"] = float(greycoprops(glcm, "contrast")[0,0])
    stats["glcm_homogeneity"] = float(greycoprops(glcm, "homogeneity")[0,0])
    stats["glcm_energy"] = float(greycoprops(glcm, "energy")[0,0])
    stats["glcm_correlation"] = float(greycoprops(glcm, "correlation")[0,0])

    # Local Binary Pattern histogram (texture)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, lbp.max() + 2),
                             range=(0, lbp.max() + 1),
                             density=True)
    stats["lbp_uniformity"] = float((hist**2).sum())

    return stats
