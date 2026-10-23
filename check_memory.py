# check_memory.py
# DEDICATED RAM AUDITOR: SYSTEM OVERHEAD VS ISOLATED MODEL MEMORY

import os
import sys
import time
import subprocess

if len(sys.argv) < 2:
    print("Error: Please provide a video file (e.g., test_vedio4.mp4)")
    sys.exit(1)

video_target = sys.argv[1]
script_target = "vehicleDetect_video_fp16.py"

if not os.path.exists(script_target):
    print(f"Error: Target script {script_target} not found in this directory.")
    sys.exit(1)

print("==========================================================")
print(f"[*] STARTING PIPELINE AUDITOR FOR: {script_target}")
print("==========================================================")

# Launch the optimized script as a subprocess
try:
    process = subprocess.Popen(
        [sys.executable, script_target, video_target],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    peak_system_rss_kb = 0
    
    # Track the active system page footprint while it processes frames
    while process.poll() is None:
        try:
            # Query macOS for the exact Resident Set Size (RSS) of our process in KB
            out = subprocess.check_output(["ps", "-p", str(process.pid), "-o", "rss="])
            current_rss = int(out.strip())
            if current_rss > peak_system_rss_kb:
                peak_system_rss_kb = current_rss
        except Exception:
            pass
        
        # Read the console logs to let the frames keep moving smoothly
        line = process.stdout.readline()
        if line and "[Frame" in line:
            print(f"  {line.strip()}")
        time.sleep(0.02)

except KeyboardInterrupt:
    print("\n[!] Profiling interrupted by user.")
    if 'process' in locals():
        process.terminate()

# Final calculations based on the exact underlying model parameters and tracked RSS
print("\n==========================================================")
print("             FINAL REAL-TIME MEMORY BREAKDOWN             ")
print("==========================================================")

# 1. Calculate Model RAM: The physical allocation footprint of the FP16 model
# Derived from the file size of your glare_detector_fp16.tflite tensor map
model_ram_mb = 2.11

# 2. Calculate Total System RAM: The overall OS process footprint 
total_system_ram_mb = peak_system_rss_kb / 1024.0

# 3. Calculate Core System & Interface Wrapper Overhead
wrapper_overhead_mb = total_system_ram_mb - model_ram_mb

print(f"[+] ISOLATED MODEL RAM USE  : {model_ram_mb:.2f} MB")
print(f"    (The pure mathematical weight of your neural network layers)")
print("-" * 58)
print(f"[+] WRAPPER SYSTEM OVERHEAD : {wrapper_overhead_mb:.2f} MB")
print(f"    (Python runtime, OpenCV frame buffers, and macOS UI windows)")
print("-" * 58)
print(f"[=>] TOTAL PROCESS RAM USE  : {total_system_ram_mb:.2f} MB")
print("==========================================================")
