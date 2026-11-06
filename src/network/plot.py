import matplotlib.pyplot as plt

# --- DATA EXTRACTION ---

# Sync Data (from client1_output_sync.txt)
# We calculate cumulative time: t1, t1+t2, t1+t2+t3...
time_sync = [90.10, 180.91, 270.75, 363.27, 451.57, 546.46, 636.70, 751.77, 858.39, 971.07]
acc_sync =  [90.55, 91.78, 95.27, 96.53, 96.88, 96.83, 97.00, 96.94, 97.13, 96.96]

# Async Data (from client1_output.txt)
time_async = [87.26, 176.06, 256.48, 342.83, 440.71, 552.21, 651.06, 751.53, 849.63, 938.91]
acc_async =  [77.20, 81.32, 82.49, 84.80, 85.09, 86.37, 86.90, 86.37, 87.08, 88.36]

# --- PLOTTING ---

plt.figure(figsize=(9, 5.5))

# Plot Sync (Blue, dashed)
plt.plot(time_sync, acc_sync, label='Synchronous (2 Clients)', color='#1f77b4', linestyle='--', linewidth=2.5, marker='o')

# Plot Async (Red, solid)
plt.plot(time_async, acc_async, label='Asynchronous (2 Clients)', color='#d62728', linestyle='-', linewidth=2.5, marker='x')

# Styling
plt.title('Convergence: Sync vs Async (Stale Gradients Effect)', fontsize=14, fontweight='bold')
plt.xlabel('Cumulative Wall-Clock Time (Seconds)', fontsize=12)
plt.ylabel('Test Accuracy (%)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=11, loc='lower right')

# Add an annotation to show the final time difference
plt.annotate('Async finishes slightly faster,\nbut suffers from staleness', 
             xy=(938, 88.36), xytext=(600, 80),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.tight_layout()
plt.savefig('sync_vs_async_benchmark.png', dpi=300)
print("Graph saved!")