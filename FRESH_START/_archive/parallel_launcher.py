
import subprocess
import time
import sys
import os

# Configuration
NUM_WORKERS = 5
PYTHON_EXE = os.path.join(".venv", "Scripts", "python.exe") # Relative path to venv

def launch_workers():
    print(f"--- Launching {NUM_WORKERS} Parallel Workers ---")
    
    processes = []
    for i in range(NUM_WORKERS):
        cmd = [PYTHON_EXE, "serial_proofread_worker.py", "--worker-id", str(i), "--total-workers", str(NUM_WORKERS)]
        # Launch independent process
        p = subprocess.Popen(cmd, cwd=os.getcwd())
        processes.append(p)
        print(f"Launched Worker {i} (PID: {p.pid})")
    
    print("All workers launched. Monitoring...")
    
    # Monitor loop
    try:
        while True:
            active = [p for p in processes if p.poll() is None]
            if not active:
                print("All workers finished!")
                break
            
            # Simple progress heartbeat
            # (Real progress is in console logs from workers)
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\nStopping all workers...")
        for p in processes:
            p.terminate()
        print("Terminated.")

if __name__ == "__main__":
    launch_workers()
