from picamera2 import Picamera2
from scipy.ndimage import median_filter
import numpy as np
import time
import threading
import queue
import os
import argparse
import signal

# ── config ─────────────────────────────────────────────────
BLACK_LEVEL   = 64
WHITE_LEVEL   = 1023
EPS           = 1e-3
QUEUE_MAXSIZE = 10
BASE_DIR      = os.path.expanduser("/home/f2e/project/sessions")
os.makedirs(BASE_DIR, exist_ok=True)

# ── args ────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="F2E Capture Pipeline")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--frames", type=int,
                      help="Capture exactly N frames. e.g. --frames 100")
    mode.add_argument("--continuous", action="store_true",
                      help="Capture until Ctrl+C.")
    mode.add_argument("--time", type=float,
                      help="Capture for N seconds. e.g. --time 10.5")

    parser.add_argument("--gain", type=float, default=16.0,
                        help="Analogue gain (default: 16.0)")
    parser.add_argument("--exposure", type=int, default=50000,
                        help="Exposure time in microseconds (default: 50000)")
    return parser.parse_args()

# ── preprocessing ───────────────────────────────────────────
def subtract_black_level(raw):
    corrected = raw.astype(np.int32) - BLACK_LEVEL
    return np.clip(corrected, 0, WHITE_LEVEL - BLACK_LEVEL).astype(np.uint16)

def split_bayer_channels(raw):
    Gb = raw[0::2, 0::2]
    B  = raw[0::2, 1::2]
    R  = raw[1::2, 0::2]
    Gr = raw[1::2, 1::2]
    return R, Gr, Gb, B

def luminance_proxy(Gr, Gb):
    return (Gr.astype(np.float32) + Gb.astype(np.float32)) / 2.0

def correct_hot_pixels(luma, threshold_sigma=5.0):
    med = median_filter(luma, size=3)
    hot_mask = (luma - med) > (threshold_sigma * luma.std())
    corrected = luma.copy()
    corrected[hot_mask] = med[hot_mask]
    return corrected

def to_log_luminance(luma):
    norm = luma / (WHITE_LEVEL - BLACK_LEVEL)
    return np.log(norm + EPS)

def preprocess_frame(raw):
    raw          = subtract_black_level(raw)
    R, Gr, Gb, B = split_bayer_channels(raw)
    luma         = luminance_proxy(Gr, Gb)
    return to_log_luminance(luma)

# ── stop condition ───────────────────────────────────────────
class StopCondition:
    def __init__(self, args):
        self.mode       = "frames" if args.frames else "time" if args.time else "continuous"
        self.max_frames = args.frames
        self.max_time   = args.time
        self.start_time = None
        self._stop      = threading.Event()

        # handle Ctrl+C gracefully
        signal.signal(signal.SIGINT, self._on_sigint)

    def start(self):
        self.start_time = time.perf_counter()

    def _on_sigint(self, *_):
        print("\nCtrl+C received — finishing up...")
        self._stop.set()

    def should_stop(self, frame_idx):
        if self._stop.is_set():
            return True
        if self.mode == "frames" and frame_idx >= self.max_frames:
            return True
        if self.mode == "time":
            elapsed = time.perf_counter() - self.start_time
            if elapsed >= self.max_time:
                return True
        return False

    def set(self):
        self._stop.set()

    def is_set(self):
        return self._stop.is_set()

# ── threads ──────────────────────────────────────────────────
CHUNK_SIZE = 50  # save every 50 frames to disk

def capture_thread(picam2, raw_queue, stop_cond, ts_buffer, session, save_dir):
    frame_idx  = 0
    chunk_idx  = 0
    chunk_raw  = []
    stop_cond.start()

    while not stop_cond.should_stop(frame_idx):
        (frame,), meta = picam2.capture_arrays(["raw"])
        raw = frame.view("uint16").copy()
        ts  = meta["SensorTimestamp"]
        ts_buffer.append(ts)
        chunk_raw.append(raw)

        try:
            raw_queue.put_nowait((frame_idx, raw, ts))
        except queue.Full:
            pass

        # flush chunk to disk
        if len(chunk_raw) >= CHUNK_SIZE:
            path = os.path.join(save_dir, f"raw_chunk{chunk_idx:04d}.npy")
            np.save(path, np.stack(chunk_raw))
            print(f"  [chunk {chunk_idx:04d}] saved {len(chunk_raw)} frames → {path}")
            chunk_raw = []
            chunk_idx += 1

        frame_idx += 1

    # save remaining frames
    if chunk_raw:
        path = os.path.join(save_dir, f"raw_chunk{chunk_idx:04d}.npy")
        np.save(path, np.stack(chunk_raw))
        print(f"  [chunk {chunk_idx:04d}] saved {len(chunk_raw)} frames → {path}")

    stop_cond.set()

def preprocess_thread(raw_queue, log_queue, stop_cond):
    while not stop_cond.is_set() or not raw_queue.empty():
        try:
            idx, raw, ts = raw_queue.get(timeout=0.5)
            log = preprocess_frame(raw)
            try:
                log_queue.put_nowait((idx, log, ts))
            except queue.Full:
                pass
        except queue.Empty:
            continue

def preview_thread(log_queue, stop_cond, session, save_dir):
    frame_count = 0
    chunk_idx   = 0
    chunk_log   = []
    t0          = time.perf_counter()

    while not stop_cond.is_set() or not log_queue.empty():
        try:
            idx, log, ts = log_queue.get(timeout=0.5)
            chunk_log.append(log)
            frame_count += 1

            if frame_count % 10 == 0:
                fps = frame_count / (time.perf_counter() - t0)
                print(f"  [frame {idx:04d}]  "
                      f"fps={fps:.1f}  "
                      f"min={log.min():.3f}  "
                      f"max={log.max():.3f}  "
                      f"mean={log.mean():.3f}")

            if len(chunk_log) >= CHUNK_SIZE:
                path = os.path.join(save_dir, f"log_chunk{chunk_idx:04d}.npy")
                np.save(path, np.stack(chunk_log))
                chunk_log = []
                chunk_idx += 1

        except queue.Empty:
            continue

    # save remaining
    if chunk_log:
        path = os.path.join(save_dir, f"{session}_log_chunk{chunk_idx:04d}.npy")
        np.save(path, np.stack(chunk_log))


# ── main ────────────────────────────────────────────────────
def main():
    args      = parse_args()
    stop_cond = StopCondition(args)
    session   = time.strftime("%Y%m%d_%H%M%S")
    SAVE_DIR = os.path.join(BASE_DIR, session)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # print mode info
    if args.frames:
        print(f"Mode      : {args.frames} frames")
    elif args.time:
        print(f"Mode      : {args.time}s duration")
    else:
        print(f"Mode      : continuous (Ctrl+C to stop)")
    print(f"Session   : {session}")
    print(f"Gain      : {args.gain}   Exposure: {args.exposure}us\n")

    # camera
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        sensor={"output_size": (1296, 972), "bit_depth": 10},
        raw={"format": "SGBRG10", "size": (1296, 972)}
    )
    picam2.configure(config)
    picam2.set_controls({
        "AeEnable": False,
        "AnalogueGain": args.gain,
        "ExposureTime": args.exposure,
        "AwbEnable": False,
        "FrameDurationLimits": (21598, 21598),
    })
    picam2.start()
    print("Camera started, warming up...")
    time.sleep(2)

    raw_queue  = queue.Queue(maxsize=QUEUE_MAXSIZE)
    log_queue  = queue.Queue(maxsize=QUEUE_MAXSIZE)
    ts_buffer  = []
    log_buffer = []

    t1 = threading.Thread(target=capture_thread,
                          args=(picam2, raw_queue, stop_cond, ts_buffer, session, SAVE_DIR))
    t2 = threading.Thread(target=preprocess_thread,
                          args=(raw_queue, log_queue, stop_cond))
    t3 = threading.Thread(target=preview_thread,
                          args=(log_queue, stop_cond, session, SAVE_DIR))

    print("Pipeline running...\n")
    t0 = time.perf_counter()
    t1.start(); t2.start(); t3.start()
    t1.join();  t2.join();  t3.join()
    picam2.stop()
    elapsed = time.perf_counter() - t0

    # save
    ts_arr  = np.array(ts_buffer, dtype=np.float64)
    np.save(os.path.join(SAVE_DIR, f"ts.npy"),  ts_arr)

    print(f"\n── Session complete ──────────────────────────────")
    print(f"  Duration  : {elapsed:.2f}s")
    print(f"  Total frames  : {len(ts_buffer)}")
    print(f"  FPS           : {len(ts_buffer) / elapsed:.2f}")
    print(f"  Saved to      : {SAVE_DIR}")

if __name__ == "__main__":
    main()
