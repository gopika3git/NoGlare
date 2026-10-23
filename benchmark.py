import cv2
import time
import numpy as np

# Define resolutions to test (Width, Height)
RESOLUTIONS = {
    "480p (SD)": (640, 480),
    "720p (HD)": (1280, 720),
    "1080p (FHD)": (1920, 1080)
}

NUM_FRAMES_TO_BENCH = 100  # Number of frames per test for a stable average

def mock_glare_processing(frame):
    """
    Placeholder representing your Glare Project logic.
    Replace this with your actual detection functions.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    return mask

def run_benchmark():
    print("=" * 60)
    print("        GLARE PROJECT: RESOLUTION VS LATENCY PROFILE        ")
    print("=" * 60)
    
    results = {}

    for res_name, (w, h) in RESOLUTIONS.items():
        print(f"\n[+] Initializing {res_name} ({w}x{h})...")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        # Request resolution from hardware
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        # Force MJPG codec to minimize high-res USB/Driver transport bottlenecks
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")) 
        
        # Warm-up frames to allow camera exposure/buffers to stabilize
        for _ in range(10):
            cap.read()
            
        latencies = []
        
        for i in range(NUM_FRAMES_TO_BENCH):
            start_time = time.perf_counter()
            
            # 1. Grab & Decode Frame
            ret, frame = cap.read()
            if not ret:
                continue
                
            # 2. Process frame (Your Glare logic)
            processed_frame = mock_glare_processing(frame)
            
            # 3. UI Display Render (Matches your macOS UI overhead)
            cv2.imshow("Glare Project Benchmark", processed_frame)
            cv2.waitKey(1)  # Minimal wait key to force redraw
            
            end_time = time.perf_counter()
            
            # Calculate total end-to-end latency for this frame (in milliseconds)
            frame_latency_ms = (end_time - start_time) * 1000
            latencies.append(frame_latency_ms)
            
        cap.release()
        cv2.destroyAllWindows()
        
        if latencies:
            avg_latency = np.mean(latencies)
            fps_equivalent = 1000 / avg_latency
            results[res_name] = {"latency": avg_latency, "fps": fps_equivalent}
            print(f"    -> Average Latency: {avg_latency:.2f} ms ({fps_equivalent:.1f} FPS)")
        else:
            print(f"    -> [ERROR] Failed to capture frames for {res_name}")

    # Print Final Summary Table
    print("\n" + "=" * 60)
    print(f"{'Resolution':<15} | {'Avg Latency (ms)':<20} | {'Est. Max FPS':<12}")
    print("-" * 60)
    for res_name, data in results.items():
        print(f"{res_name:<15} | {data['latency']:<20.2f} | {data['fps']:<12.1f}")
    print("=" * 60)

if __name__ == "__main__":
    run_benchmark()
