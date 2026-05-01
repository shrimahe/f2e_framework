from picamera2 import Picamera2
import numpy as np
import time

picam2 = Picamera2()

config = picam2.create_video_configuration(
    sensor={"output_size": (1296, 972), "bit_depth": 10},
    raw={"format": "SGBRG10", "size": (1296, 972)}
)
picam2.configure(config)

picam2.set_controls({
    "AeEnable": False,
    "AnalogueGain": 16.0,
    "ExposureTime": 50000,
    "AwbEnable": False,
    "FrameDurationLimits": (21598, 21598),
})

picam2.start()
print("Camera started, warming up...")
time.sleep(2)

N = 100
frames = []
timestamps = []
t0 = time.perf_counter()

for i in range(N):
    (frame,), meta = picam2.capture_arrays(["raw"])
    frames.append(frame.view("uint16").copy())
    timestamps.append(meta["SensorTimestamp"])
    if (i + 1) % 10 == 0:
        print(f"  Captured {i+1}/{N} frames")

elapsed = time.perf_counter() - t0
picam2.stop()

frames = np.stack(frames)
timestamps = np.array(timestamps, dtype=np.float64)

dt = np.diff(timestamps) / 1e6  # nanoseconds to milliseconds

print(f"\nResults:")
print(f"  Frames     : {frames.shape}")
print(f"  FPS        : {N / elapsed:.2f}")
print(f"  Interval   : mean={dt.mean():.2f}ms  std={dt.std():.2f}ms")
print(f"  Pixel min  : {frames.min()}")
print(f"  Pixel max  : {frames.max()}")
print(f"  Pixel mean : {frames.mean():.1f}")

np.save("raw_frames.npy", frames)
np.save("timestamps.npy", timestamps)
print(f"\nSaved: raw_frames.npy {frames.shape}")
print(f"Saved: timestamps.npy {timestamps.shape}")
