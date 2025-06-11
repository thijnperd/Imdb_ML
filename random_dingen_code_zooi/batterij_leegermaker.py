import threading
import time
import os

def burn_cpu():
    while True:
        x = 0
        for i in range(1000000):
            x += i**0.5

def main():
    num_threads = os.cpu_count() or 4
    print(f"Spinning up {num_threads} threads to heat up your CPU...")

    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=burn_cpu)
        t.daemon = True
        t.start()
        threads.append(t)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped CPU stress test.")

if __name__ == "__main__":
    main()
