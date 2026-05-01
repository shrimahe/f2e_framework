import numpy as np

# ── constants ──────────────────────────────────────────────
BLACK_LEVEL = 64
WHITE_LEVEL = 1023
EPS         = 1e-3

# ── 2.1 black level ────────────────────────────────────────
def subtract_black_level(raw):
    corrected = raw.astype(np.int32) - BLACK_LEVEL
    return np.clip(corrected, 0, WHITE_LEVEL - BLACK_LEVEL).astype(np.uint16)

# ── 2.2 channel split (GBRG pattern) ───────────────────────
def split_bayer_channels(raw):
    Gb = raw[0::2, 0::2]
    B  = raw[0::2, 1::2]
    R  = raw[1::2, 0::2]
    Gr = raw[1::2, 1::2]
    return R, Gr, Gb, B

def luminance_proxy(Gr, Gb):
    return (Gr.astype(np.float32) + Gb.astype(np.float32)) / 2.0

# ── 2.3 hot pixel correction ────────────────────────────────
from scipy.ndimage import median_filter

def correct_hot_pixels(luma, threshold_sigma=5.0):
    med = median_filter(luma, size=3)
    diff = luma - med
    hot_mask = diff > (threshold_sigma * luma.std())
    corrected = luma.copy()
    corrected[hot_mask] = med[hot_mask]
    return corrected

# ── 2.5 log luminance ───────────────────────────────────────
def to_log_luminance(luma):
    norm = luma / (WHITE_LEVEL - BLACK_LEVEL)
    return np.log(norm + EPS)

# ── full pipeline ───────────────────────────────────────────
def preprocess_frame(raw):
    raw  = subtract_black_level(raw)
    R, Gr, Gb, B = split_bayer_channels(raw)
    luma = luminance_proxy(Gr, Gb)
    luma = correct_hot_pixels(luma)
    log  = to_log_luminance(luma)
    return log

# ── main ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading raw frames...")
    frames = np.load("raw_frames.npy")
    print(f"Loaded: {frames.shape} {frames.dtype}")

    N = frames.shape[0]
    processed = []

    for i in range(N):
        log_frame = preprocess_frame(frames[i])
        processed.append(log_frame)
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{N}")

    processed = np.stack(processed)

    print(f"\nResults:")
    print(f"  Output shape : {processed.shape}")
    print(f"  Dtype        : {processed.dtype}")
    print(f"  Min          : {processed.min():.4f}")
    print(f"  Max          : {processed.max():.4f}")
    print(f"  Mean         : {processed.mean():.4f}")

    np.save("logluma_frames.npy", processed)
    print(f"\nSaved: logluma_frames.npy {processed.shape}")
