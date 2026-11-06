import subprocess
import time

def run_benchmark():
    server_log = open("server_output_sync.txt", "w")
    server_process = subprocess.Popen(["python3", "src/network/server_sync.py"], stdout=server_log, stderr=subprocess.STDOUT)
    
    time.sleep(2) 
    
    c1_log = open("client1_output_sync.txt", "w")
    client1 = subprocess.Popen(["python3", "src/network/client_sync.py"], stdout=c1_log, stderr=subprocess.STDOUT)
    
    c2_log = open("client2_output_sync.txt", "w")
    client2 = subprocess.Popen(["python3", "src/network/client_sync.py"], stdout=c2_log, stderr=subprocess.STDOUT)
    
    start_time = time.perf_counter()
    
    client1.wait()
    print("Client 1 Finished. (Check client1_output_sync.txt for loss)")
    client2.wait()
    print("Client 2 Finished. (Check client2_output_sync.txt for loss)")
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    server_process.terminate()
    server_log.close()
    c1_log.close()
    c2_log.close()
    
    print("====================================")
    print(f"TOTAL SYNC TRAINING TIME: {total_time:.2f} seconds")
    print("====================================")

if __name__ == "__main__":
    run_benchmark()