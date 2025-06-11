import threading
import time
import os
import psutil
import sys

# CPU load function
def burn_cpu():
    while True:
        _ = sum(i**0.5 for i in range(1000000))

# Battery monitor
def monitor_battery():
    while True:
        battery = psutil.sensors_battery()
        if battery is None:
            print("Battery info not available. Exiting...")
            os._exit(1)

        percent = battery.percent
        plugged = battery.power_plugged

        print(f"Battery: {percent:.1f}% {'(Charging)' if plugged else '(Discharging)'}")
        
        if not plugged and percent <= 10:
            print("Battery at or below 10%. Stopping load.")
            os._exit(0)  # Immediately stops all threads
        
        time.sleep(10)  # Check every 10 seconds

# Main function
def main():
    if not hasattr(psutil, "sensors_battery"):
        print("psutil doesn't support battery monitoring on this system.")
        sys.exit(1)

    # Start CPU threads
    num_threads = os.cpu_count() or 4
    print(f"Starting {num_threads} CPU threads to drain battery...")

    for _ in range(num_threads):
        t = threading.Thread(target=burn_cpu)
        t.daemon = True
        t.start()

    # Start battery monitoring
    monitor_battery()

if __name__ == "__main__":
    main()
