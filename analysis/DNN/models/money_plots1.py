import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.patches as mpatches
import sys

z_size = 12.
x_size = 3.
y_size = 3.

analysis_type = sys.argv[1] # ppi = proton, pion , pik = pion, kaon , pk = proton, kaon , ppik = proton, pion, kaon

folder_path = "../../results/xgboost/"
if analysis_type == "ppi":
    folder_path += "proton_pion"
elif analysis_type == "pik":
    folder_path += "pion_kaon"
elif analysis_type == "pk":
    folder_path += "proton_kaon"
elif analysis_type == "ppik":
    folder_path += "proton_pion_kaon"

target_file = "/home/alma1/GNN/Deepset/28oct_DNN/TowardPIDwithGranularCalorimeters/results/DNN/accuracyTable.csv"
summaryTable = pd.read_csv(target_file, sep=",")
summaryTable = summaryTable[summaryTable["accuracy"] != -1]


summaryTable[["segx", "segy", "segz"]] = summaryTable["segmentation"].str.extract(r"(\d+)_(\d+)_(\d+)")
summaryTable["segx"] = summaryTable["segx"].astype(float)
summaryTable["segy"] = summaryTable["segy"].astype(float)
summaryTable["segz"] = summaryTable["segz"].astype(float)

segZ_accuracy = (
    summaryTable.sort_values(by=["segz", "segx"])  
    .groupby("segz")["accuracy"]         
    .apply(list)                         
    .to_dict()                           
)

segZ_minAcc = (
    summaryTable.sort_values(by=["segz", "segx"])  
    .groupby("segz")["minVal"]         
    .apply(list)                         
    .to_dict()                           
)

segZ_maxAcc = (
    summaryTable.sort_values(by=["segz", "segx"])  
    .groupby("segz")["maxVal"]         
    .apply(list)                         
    .to_dict()                           
)

segXY_accuracy = (
    summaryTable.sort_values(by=["segx", "segz"])  
    .groupby("segx")["accuracy"]         
    .apply(list)                         
    .to_dict()                          
)

segXY_minAcc = (
    summaryTable.sort_values(by=["segx", "segz"])  
    .groupby("segx")["minVal"]         
    .apply(list)                         
    .to_dict()                           
)

segXY_maxAcc = (
    summaryTable.sort_values(by=["segx", "segz"])  
    .groupby("segx")["maxVal"]         
    .apply(list)                         
    .to_dict()                           
)

volume_XYZ = ((100/summaryTable["segx"]) * (100/summaryTable["segy"])*(100/summaryTable["segz"])*x_size*y_size*z_size).tolist()
accuracy_XYZ = summaryTable["accuracy"].tolist()
error_XYZ_min = summaryTable["minVal"].tolist()
error_XYZ_max = summaryTable["maxVal"].tolist()

error_XYZ_lower = [accuracy - error_min for accuracy, error_min in zip(accuracy_XYZ, error_XYZ_min)]
error_XYZ_upper = [error_max - accuracy for accuracy, error_max in zip(accuracy_XYZ, error_XYZ_max)]

volume_XYZ, accuracy_XYZ, error_XYZ_lower, error_XYZ_upper = zip(*sorted(zip(volume_XYZ, accuracy_XYZ, error_XYZ_lower, error_XYZ_upper), key=lambda x: x[0]))


baseline_file = "/home/alma1/GNN/Deepset/28oct_DNN/TowardPIDwithGranularCalorimeters/results/DNN/baseline.tsv"
baselineTable = pd.read_csv(baseline_file, sep="\t")
baselineTable['accuracy']=baselineTable['accuracy']*100.
baselineTable['minVal']=baselineTable['minVal']*100.
baselineTable['maxVal']=baselineTable['maxVal']*100.
base = baselineTable['accuracy'].iloc[0].astype(float)
base_min = baselineTable['minVal'].iloc[0].astype(float)
base_max = baselineTable['maxVal'].iloc[0].astype(float)

# Define the color for the 1-sigma region
sigma_color = 'red'

# Create the figure with 2 rows and 2 columns
fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharey=False)
axes[0, 0].tick_params(axis='both', which='major', labelsize=14)
axes[0, 1].tick_params(axis='both', which='major', labelsize=14)
axes[1, 0].tick_params(axis='both', which='major', labelsize=14)
axes[1, 1].tick_params(axis='both', which='major', labelsize=14)


# Plot 1: Accuracy vs Area_XY
for segz in segZ_accuracy.keys():
    # Get the accuracy, minVal, and maxVal for the current segx
    accuracy = segZ_accuracy[segz]
    min_val = segZ_minAcc[segz]
    max_val = segZ_maxAcc[segz]
    
    # Calculate the error (difference between accuracy and min/max)
    error = [accuracy - min_v for accuracy, min_v in zip(accuracy, min_val)]
    error_upper = [max_v - accuracy for max_v, accuracy in zip(max_val, accuracy)]

    areaXY = [100 / segx * 100 / segx * x_size * y_size for segx in [10.0, 25.0, 50.0, 100.0]]
    
    # Plot the curve with error bars
    axes[0, 0].errorbar(
        areaXY,                     
        accuracy, 
        yerr=[error, error_upper],  # Lower and upper error
        label=f'segZ = {segz}',     # Legend for each segx
        fmt='--o',                  # Line style and marker
        capsize=5                   # Add caps to the error bars
    )
    
axes[0, 0].axhspan(
    base_min, base_max, color=sigma_color, alpha=0.3, label=r'Baseline $\sigma=1$'
)


if analysis_type == "ppi":
    axes[0, 0].set_title(r'Accuracy $p/\pi$ as a Function of Cell Cross-Section Area', fontsize=14) 
elif analysis_type == "pik":
    axes[0, 0].set_title(r'Accuracy $\pi/\K$ as a Function of Cell Cross-Section Area', fontsize=14) 
elif analysis_type == "pk":
    axes[0, 0].set_title(r'Accuracy $p/\K$ as a Function of Cell Cross-Section Area', fontsize=14) 

axes[0, 0].axhline(y=base, color='red', linestyle='--', label='Baseline')  # Add baseline
axes[0, 0].set_xlabel(r'$\Delta_{XY} \,\, [\mathrm{mm}^2]$', fontsize=14)
axes[0, 0].set_ylabel('Accuracy', fontsize=14)
axes[0, 0].legend()  # Show the legend
axes[0, 0].grid(True, linestyle='--', alpha=0.7)
axes[0, 0].set_xscale('log')

# Plot 2: Accuracy vs Delta_Z
for segx in segXY_accuracy.keys():
    # Get the accuracy, minVal, and maxVal for the current segx
    accuracy = segXY_accuracy[segx]
    min_val = segXY_minAcc[segx]
    max_val = segXY_maxAcc[segx]
    
    # Calculate the error (difference between accuracy and min/max)
    error = [accuracy - min_v for accuracy, min_v in zip(accuracy, min_val)]
    error_upper = [max_v - accuracy for max_v, accuracy in zip(max_val, accuracy)]

    # Calculate deltaZ (deltaZ = 100 / segz * z_size)
    deltaZ = [100 / segz * z_size for segz in [10.0, 25.0, 50.0, 100.0]] 

    # Plot the curve with error bars
    axes[0, 1].errorbar(
        deltaZ,  # Use deltaZ for the x-axis
        accuracy, 
        yerr=[error, error_upper],  # Lower and upper error
        label=f'segXY = {segx}',     # Legend for each segx
        fmt='--o',                  # Line style and marker
        capsize=5                   # Add caps to the error bars
    )

axes[0, 1].axhspan(
    base_min, base_max, color=sigma_color, alpha=0.3, label=r'Baseline $\sigma=1$'
)

if analysis_type == "ppi":
    axes[0, 1].set_title(r'Accuracy $p/\pi$ as a Function of Longitudinal Segmentation', fontsize=14) 
elif analysis_type == "pik":
    axes[0, 1].set_title(r'Accuracy $\pi/\K$ as a Function of Longitudinal Segmentation', fontsize=14) 
elif analysis_type == "pk":
    axes[0, 1].set_title(r'Accuracy $p/\K$ as a Function of Longitudinal Segmentation', fontsize=14) 
    
axes[0, 1].axhline(y=base, color='red', linestyle='--', label='Baseline') 
axes[0, 1].set_xlabel(r'$\Delta_{Z} \,\, [\mathrm{mm}]$', fontsize=14)
axes[0, 1].grid(True, linestyle='--', alpha=0.7)
axes[0, 1].legend()  # Show the legend
axes[0, 1].set_xscale('log')



# Plot 3: Accuracy vs Volume_XYZ
axes[1, 0].errorbar(
    volume_XYZ, accuracy_XYZ, 
    yerr=(error_XYZ_lower, error_XYZ_upper), 
    fmt='s--', capsize=5, label='Accuracy', color='green'
)
axes[1, 0].axhspan(
    base_min, base_max, color=sigma_color, alpha=0.3, label=r'Baseline $\sigma=1$'
)

if analysis_type == "ppi":
    axes[1, 0].set_title(r'Accuracy $p/\pi$ as a Function of Cell Volume', fontsize=14) 
elif analysis_type == "pik":
    axes[1, 0].set_title(r'Accuracy $\pi/\K$ as a Function of Cell Volume', fontsize=14) 
elif analysis_type == "pk":
    axes[1, 0].set_title(r'Accuracy $p/\K$ as a Function of Cell Volume', fontsize=14) 
    
axes[1, 0].axhline(y=base, color='red', linestyle='--', label='Baseline') 
axes[1, 0].set_xlabel(r'$\Delta_{XYZ} \,\, [\mathrm{mm}^3]$', fontsize=14)
axes[1, 0].set_ylabel('Accuracy', fontsize=14)
axes[1, 0].grid(True, linestyle='--', alpha=0.7)
axes[1, 0].legend()
axes[1, 0].set_xscale('log')


# Plot 4: Matrix of Accuracy (seg_xy vs seg_z)
# Create an accuracy matrix with seg_x as x and seg_z as y
seg_xy_values = np.array([100,50,25,10])
seg_z_values = np.array([100,50,25,10])

accuracy_matrix = np.zeros((len(seg_z_values), len(seg_xy_values)))

for i, seg_x in enumerate(seg_xy_values):
    for j, seg_z in enumerate(seg_z_values):
        # Find the average accuracy for each combination of seg_x and seg_z
        subset = summaryTable[(summaryTable["segx"] == seg_x) & (summaryTable["segz"] == seg_z)]
        
        if (len(subset)):
            accuracy_matrix[j, i] = subset["accuracy"].mean()
        else:
            accuracy_matrix[j, i] = 0.

labels_x = ((100)/(seg_xy_values))*((100)/(seg_xy_values))*x_size*y_size
labels_y = 100/seg_z_values*z_size

heatmap_kwargs = {
    'annot': True,
    'fmt': ".3f",
    'cmap': 'Blues',
    'vmin': 60.00,
    'vmax': 63.00, 
    'annot_kws': {"size": 16},
    'cbar': True 
}

import seaborn as sns
print(f"accuracy_matrix: {accuracy_matrix}")
sns.heatmap(
    accuracy_matrix, 
    xticklabels=labels_x, 
    yticklabels=labels_y, 
    ax=axes[1, 1],
    **heatmap_kwargs
)

axes[1, 1].set_xlabel(r'$\Delta_{XY} \,\, [\mathrm{mm}^2]$', fontsize=14)
axes[1, 1].set_ylabel(r'$\Delta_{Z} \,\, [\mathrm{mm}]$', fontsize=14)
axes[1, 1].set_title('Accuracy Matrix', fontsize=14)

if analysis_type == "ppi":
    axes[1, 1].set_title(r'Accuracy $p/\pi$ as a Function of Segmentation', fontsize=14) 
elif analysis_type == "pik":
    axes[1, 1].set_title(r'Accuracy $\pi/\K$ as a Function of Segmentation', fontsize=14) 
elif analysis_type == "pk":
    axes[1, 1].set_title(r'Accuracy $p/\K$ as a Function of Segmentation', fontsize=14) 

plt.tight_layout()
plt.savefig(f'/home/alma1/GNN/Deepset/28oct_DNN/TowardPIDwithGranularCalorimeters/results/DNN/moneyplot.png')
plt.savefig(f'/home/alma1/GNN/Deepset/28oct_DNN/TowardPIDwithGranularCalorimeters/results/DNN/moneyplot.pdf')
