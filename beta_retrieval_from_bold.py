#%%
import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.input_data import NiftiLabelsMasker
from nilearn.signal import clean
from nilearn import image
from nilearn import datasets
import nibabel as nib
import numpy as np
import pandas as pd
import numpy as np

#%%
# Base path
fmri_base_path = 'flux-data/'
output_dir = 'output/'
type_data = 'denoised'  #denoised or interpolated choose the different image you want to use
N_networks = 7
confounds_choice = [
                "trans_x", "trans_y", "trans_z",
                "rot_x", "rot_y", "rot_z",
                "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
                "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1",
                "trans_x_power2", "trans_y_power2", "trans_z_power2",
                "rot_x_power2", "rot_y_power2", "rot_z_power2",
                "trans_x_derivative1_power2", "trans_y_derivative1_power2", "trans_z_derivative1_power2",
                "rot_x_derivative1_power2", "rot_y_derivative1_power2", "rot_z_derivative1_power2"
            ]

# Get all participant directories starting with "sub-"
participants = [d for d in os.listdir(fmri_base_path) if d.startswith('sub-') and os.path.isdir(os.path.join(fmri_base_path, d))]
print(f"Found {len(participants)} participants: {participants}")

os.makedirs(output_dir, exist_ok=True)

#%%
# Load Yeo 7-network atlas
print("Loading Yeo 7-network atlas...")
atlas_yeo = datasets.fetch_atlas_yeo_2011(n_networks=N_networks, thickness='thick')
atlas_img = nib.load(atlas_yeo.maps)  # Nifti image of atlas
atlas_data = atlas_img.get_fdata().astype(int)

# Handle 4D atlas (sometimes singleton dimension)
if atlas_data.ndim == 4 and atlas_data.shape[3] == 1:
    atlas_data = np.squeeze(atlas_data, axis=3)

unique_labels = np.unique(atlas_data)
unique_labels = unique_labels[unique_labels != 0]  # exclude background

# Get network names
network_names = atlas_yeo.labels[1:]  # skip background label

print(f"Atlas shape: {atlas_data.shape}")
print(f"Atlas networks: {network_names}")

#%%
for participant_id in participants:
    participant_path = os.path.join(fmri_base_path, 'derivatives', 'fmriprep-23.1.3', participant_id, 'func') #change this if you organise it in another way


    # Find fMRI file
    if type_data == 'denoised':
        fmri_files = glob.glob(os.path.join(participant_path, '*_desc-preproc_bold.nii*'))
    elif type_data == 'interpolated':
        fmri_files = glob.glob(os.path.join(participant_path, '*_desc-interpolated_bold.nii*'))

    if len(fmri_files) == 0:
        print(f"No fMRI file found for {participant_id}, skipping...")
        continue

    fmri_file = fmri_files[0]  # take the first match
    output_file = os.path.join(output_dir, f"{participant_id}_connectivity_matrix.csv")

    if os.path.exists(output_file):
        print(f"Skipping participant {participant_id}: {output_file} already exists.")
        continue

    try:
        fmri_img = nib.load(fmri_file)

        if type_data == 'interpolated':
            # Find TSV confound file
            tsv_files = glob.glob(os.path.join(participant_path, '*_desc_counfounds_timeseries.tsv'))
            if len(tsv_files) == 0:
                print(f"No confound TSV file found for {participant_id}, skipping...")
                continue
            confounds_file = tsv_files[0]

            confounds = pd.read_csv(confounds_file, sep='\t')
            confound_columns_24P = confounds_choice
            confound_vars = confounds[confound_columns_24P].fillna(0).values

            # Denoise
            func_data = fmri_img.get_fdata()
            func_data_2d = func_data.reshape(-1, func_data.shape[-1]).T
            cleaned_data = clean(func_data_2d, confounds=confound_vars, detrend=True, standardize=True)
            cleaned_4d = cleaned_data.T.reshape(fmri_img.shape)
            fmri_img = image.new_img_like(fmri_img, cleaned_4d)

        # Extract time series & compute connectivity
        masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=True)
        time_series = masker.fit_transform(fmri_img)
        correlation_matrix = np.corrcoef(time_series.T)
        correlation_df = pd.DataFrame(correlation_matrix, index=network_names, columns=network_names)
        correlation_df.to_csv(output_file, index=True)
        print(f"Processed participant {participant_id}, matrix saved to {output_file}")

    except Exception as e:
        print(f"Error processing participant {participant_id}: {e}")

# %%

all_data = []  # store upper-triangle values
all_participants = []

n_networks = len(network_names)

# Loop over saved connectivity matrices
for participant_id in participants:
    matrix_file = os.path.join(output_dir, f"{participant_id}_connectivity_matrix.csv")
    if not os.path.exists(matrix_file):
        continue

    corr_df = pd.read_csv(matrix_file, index_col=0)
    corr_matrix = corr_df.values

    # Get upper triangle indices (excluding diagonal)
    triu_indices = np.triu_indices(n_networks, k=1)
    upper_tri_values = corr_matrix[triu_indices]

    all_data.append(upper_tri_values)
    all_participants.append(participant_id)

# Generate column labels dynamically from atlas labels
column_labels = [f"{network_names[i]}-{network_names[j]}" 
                 for i, j in zip(*np.triu_indices(n_networks, k=1))]

# Create final DataFrame
fc_tabular = pd.DataFrame(all_data, index=all_participants, columns=column_labels)
fc_tabular.to_csv(os.path.join(output_dir, 'FC_tabular.csv'))
print("Tabular functional connectivity saved as 'FC_tabular.csv'")

# %%
