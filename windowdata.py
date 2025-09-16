import wfdb
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# Step 1: Label mapping for selected beat types
label_map = {
    'N': 'Normal',
    'L': 'LBBB',
    'R': 'RBBB',
    'A': 'Atrial Premature',
    'V': 'PVC',
    'F': 'Fusion',
    'E': 'Atrial Escape',
}

# Step 2: MIT-BIH official record IDs (48 standard records)
record_ids = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119",
    "121", "122", "123", "124", "200", "201", "202", "203", "205",
    "207", "208", "209", "210", "212", "213", "214", "215", "217",
    "219", "220", "221", "222", "223", "228", "230", "231", "232", "233", "234"
]

# Step 3: Create download directory and download files if needed
download_dir = 'mitdb'
if not os.path.exists(os.path.join(download_dir, '100.dat')):
    print("📥 Downloading MIT-BIH selected records...")
    wfdb.dl_database('mitdb', dl_dir=download_dir, records=record_ids)
else:
    print("✅ MIT-BIH files already exist locally.")

# Step 4: Multi-resolution window setup
window_sizes = [180, 360, 540]
half_sizes = [w // 2 for w in window_sizes]
max_half_size = max(half_sizes)

# Store all samples
all_data = []

# Step 5: Debug-enabled record processing loop
for record_id in tqdm(record_ids, desc="🔄 Processing Records"):
    record_path = os.path.join(download_dir, record_id)

    print(f"\n📂 Now processing: {record_id}")
    
    try:
        # Read signal
        try:
            record = wfdb.rdrecord(record_path)
            print(f"✅ Signal loaded for {record_id}")
        except Exception as e:
            print(f"❌ Signal load failed for {record_id}: {e}")
            continue

        # Read annotations
        try:
            annotation = wfdb.rdann(record_path, 'atr')
            print(f"✅ Annotations loaded for {record_id}")
        except Exception as e:
            print(f"❌ Annotation load failed for {record_id}: {e}")
            continue

        ecg_signal = record.p_signal[:, 0]
        n_samples = len(ecg_signal)
        print(f"ℹ️ ECG signal length: {n_samples} samples")

        # Process annotated beats
        for sample_index, symbol in zip(annotation.sample, annotation.symbol):
            if symbol not in label_map:
                continue
            if (sample_index - max_half_size < 0) or (sample_index + max_half_size >= n_samples):
                continue

            label = label_map[symbol]
            features = []

            for w in window_sizes:
                half_w = w // 2
                segment = ecg_signal[sample_index - half_w : sample_index + half_w]
                features.extend(segment)

            features.append(label)
            all_data.append(features)

    except Exception as general_error:
        print(f"⚠️ Unexpected error in {record_id}: {general_error}")

# Step 6: Create DataFrame and save
column_names = [f'ECG_Sample_{i}' for i in range(sum(window_sizes))] + ['Label']
df = pd.DataFrame(all_data, columns=column_names)

df.to_csv('ecg_multi_resolution_all.csv', index=False)
print("\n✅ File saved: ecg_multi_resolution_all.csv")
print("📐 Final dataset shape:", df.shape)
print("\n📊 Class-wise summary:")
print(df['Label'].value_counts())
