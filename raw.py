import wfdb
import numpy as np
import pandas as pd



ecg_signal = record.p_signal[:, 0]
n_samples = len(ecg_signal)

# Define the mapping from annotation symbols to class labels
label_map = {
    'N': 'Normal',
    'L': 'LBBB',
    'R': 'RBBB',
    'A': 'Atrial Premature',
    'V': 'PVC',
    'F': 'Fusion',
    'E': 'Atrial Escape',
}

# Define window sizes for multi-resolution sliding
window_sizes = [180, 360, 540]  # Short, Medium, Long
half_sizes = [w // 2 for w in window_sizes]
max_half_size = max(half_sizes)

# Prepare dataset
multi_res_data = []

for sample_index, symbol in zip(annotation.sample, annotation.symbol):
    if symbol not in label_map:
        continue  # skip unknown or unwanted beats
    
    label = label_map[symbol]

    # Check if signal has enough padding for all window sizes
    if (sample_index - max_half_size < 0) or (sample_index + max_half_size >= n_samples):
        continue

    sample_features = []

    for w in window_sizes:
        half_w = w // 2
        start = sample_index - half_w
        end = sample_index + half_w
        window = ecg_signal[start:end]
        sample_features.extend(window)  # flatten and concatenate

    sample_features.append(label)  # Add class label
    multi_res_data.append(sample_features)

# Column names
feature_cols = [f'ECG_Sample_{i}' for i in range(sum(window_sizes))]
df = pd.DataFrame(multi_res_data, columns=feature_cols + ['Label'])

# Save to CSV
df.to_csv('ecg_multi_resolution_100.csv', index=False)
print("✅ File saved as: ecg_multi_resolution_100.csv")
print("📐 Shape:", df.shape)

# Summary: class-wise count
summary = df['Label'].value_counts().reset_index()
summary.columns = ['Class Label', 'Count']
print("\n📊 Class-wise Summary of Extracted Beats:")
print(summary)


