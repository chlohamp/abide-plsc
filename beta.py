import os
import glob
import re
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets
from nilearn.image import resample_to_img

# -------------------------------
# USER PARAMETERS
# -------------------------------
input_dir = 'flux-data/'
file_pattern = '*_conn.nii.gz'   # look for {sub_id}_conn.nii.gz
output_file = 'beta_network_matrix.csv'

# -------------------------------
# FETCH YEO 7-NETWORK ATLAS
# -------------------------------
print("Loading Yeo 7-network atlas...")
atlas_yeo = datasets.fetch_atlas_yeo_2011()
atlas_file = atlas_yeo.thick_7   # 7-network parcellation
atlas_img = nib.load(atlas_file)
atlas_data = atlas_img.get_fdata().astype(int)

# Handle 4D atlas (squeeze out singleton dimension)
if atlas_data.ndim == 4:
    print(f"Atlas is 4D: {atlas_data.shape}")
    if atlas_data.shape[3] == 1:
        atlas_data = np.squeeze(atlas_data, axis=3)
        print(f"Squeezed atlas to 3D: {atlas_data.shape}")
    else:
        raise ValueError(f"Atlas has {atlas_data.shape[3]} volumes, expected 1")

unique_labels = np.unique(atlas_data)
unique_labels = unique_labels[unique_labels != 0]  # exclude background

print(f"Final atlas shape: {atlas_data.shape}")
print(f"Atlas unique labels: {unique_labels}")

# Yeo-7 network names (order matches atlas labels 1–7)
network_names = [
    "Visual",
    "Somatomotor", 
    "DorsalAttention",
    "VentralAttention",
    "Limbic",
    "Frontoparietal",
    "Default"
]

if len(unique_labels) != len(network_names):
    raise ValueError(f"Number of atlas labels ({len(unique_labels)}) does not match expected Yeo 7 networks ({len(network_names)}).")

# -------------------------------
# FIND SUBJECT FILES
# -------------------------------
search_pattern = os.path.join(input_dir, '**', file_pattern)
conn_files = sorted(glob.glob(search_pattern, recursive=True))
print(f"\nFound {len(conn_files)} *_conn.nii.gz files:")
for f in conn_files[:5]:  # Show first 5 files
    print(f"  {f}")
if len(conn_files) > 5:
    print(f"  ... and {len(conn_files) - 5} more")

if len(conn_files) == 0:
    print("ERROR: No files found! Check your input_dir and file_pattern.")
    exit(1)

# -------------------------------
# PROCESS ALL SUBJECTS
# -------------------------------
all_subjects_data = []
failed_subjects = []

for i, conn_file in enumerate(conn_files):
    filename = os.path.basename(conn_file)
    print(f"\n[{i+1}/{len(conn_files)}] Processing: {filename}")
    
    # Extract subject ID
    match = re.search(r'(sub-[^_]+)', filename)
    if not match:
        print(f"  ❌ Cannot extract subject ID from filename, skipping.")
        failed_subjects.append(filename)
        continue
    subject_id = match.group(1)
    print(f"  Subject ID: {subject_id}")

    try:
        # Load subject data
        conn_img = nib.load(conn_file)
        conn_data = conn_img.get_fdata()
        
        print(f"  Data shape: {conn_data.shape}")
        print(f"  Data range: [{np.min(conn_data):.4f}, {np.max(conn_data):.4f}]")
        print(f"  Data dtype: {conn_data.dtype}")
        
        # Check for NaN/inf values
        nan_count = np.sum(np.isnan(conn_data))
        inf_count = np.sum(np.isinf(conn_data))
        if nan_count > 0 or inf_count > 0:
            print(f"  ⚠️  WARNING: Found {nan_count} NaN and {inf_count} inf values")

        # Handle 4D images (common with singleton 4th dimension)
        if conn_data.ndim == 4:
            print(f"  📦 4D image detected, shape: {conn_data.shape}")
            if conn_data.shape[3] == 1:
                # Squeeze out singleton dimension
                conn_data = np.squeeze(conn_data, axis=3)
                print(f"  ✅ Squeezed to 3D: {conn_data.shape}")
            else:
                print(f"  ❌ ERROR: 4D image has {conn_data.shape[3]} volumes, expected 1. Skipping...")
                failed_subjects.append(filename)
                continue
        elif conn_data.ndim != 3:
            print(f"  ❌ ERROR: Expected 3D map, got {conn_data.ndim}D shape {conn_data.shape}, skipping...")
            failed_subjects.append(filename)
            continue

        # Check if dimensions match atlas (both should now be 3D)
        if conn_data.shape != atlas_data.shape:
            print(f"  ⚠️  WARNING: Data shape {conn_data.shape} != atlas shape {atlas_data.shape}")
            print("  Attempting to resample data to match atlas...")
            
            try:
                # Create a temporary 3D atlas image for resampling
                atlas_img_3d = nib.Nifti1Image(atlas_data, atlas_img.affine, atlas_img.header)
                conn_img_resampled = resample_to_img(conn_img, atlas_img_3d, interpolation='linear')
                conn_data = conn_img_resampled.get_fdata()
                print(f"  ✅ Resampled to shape: {conn_data.shape}")
            except Exception as e:
                print(f"  ❌ ERROR: Failed to resample: {e}")
                failed_subjects.append(filename)
                continue

        # Compute mean beta per Yeo network
        network_betas = []
        for j, label in enumerate(unique_labels):
            mask = atlas_data == label
            voxel_count = np.sum(mask)
            
            if voxel_count == 0:
                print(f"    Network {network_names[j]} (label {label}): No voxels found!")
                network_betas.append(np.nan)
                continue
                
            # Extract values and compute mean, excluding NaN/inf
            network_values = conn_data[mask]
            valid_values = network_values[np.isfinite(network_values)]
            
            if len(valid_values) == 0:
                print(f"    Network {network_names[j]} (label {label}): No valid values!")
                mean_beta = np.nan
            else:
                mean_beta = np.mean(valid_values)
            
            network_betas.append(mean_beta)
            print(f"    Network {network_names[j]} (label {label}): {voxel_count} voxels, mean = {mean_beta:.6f}")

        all_subjects_data.append([subject_id] + network_betas)
        print(f"  ✅ Successfully processed {subject_id}")
        
    except Exception as e:
        print(f"  ❌ ERROR processing {filename}: {e}")
        failed_subjects.append(filename)
        continue

# -------------------------------
# SAVE GROUP CSV
# -------------------------------
if len(all_subjects_data) == 0:
    print("\n❌ ERROR: No subjects were successfully processed!")
    exit(1)

columns = ["subject_id"] + network_names
df = pd.DataFrame(all_subjects_data, columns=columns)

print(f"\n📊 SUMMARY:")
print(f"  Successfully processed: {len(all_subjects_data)} subjects")
print(f"  Failed: {len(failed_subjects)} subjects")
if failed_subjects:
    print(f"  Failed files: {failed_subjects}")

print(f"\n📋 Data preview:")
print(df.head())
print(f"\n📊 Data statistics:")
print(df.describe())

# Save to CSV
df.to_csv(output_file, index=False)
print(f"\n✅ Saved group beta matrix: {output_file}")

# Additional validation
print(f"\n🔍 VALIDATION:")
print(f"  Output file size: {os.path.getsize(output_file)} bytes")
print(f"  CSV shape: {df.shape}")

# Check for any completely missing networks
for col in network_names:
    missing_count = df[col].isna().sum()
    if missing_count > 0:
        print(f"  ⚠️  {col}: {missing_count}/{len(df)} subjects have missing values")

