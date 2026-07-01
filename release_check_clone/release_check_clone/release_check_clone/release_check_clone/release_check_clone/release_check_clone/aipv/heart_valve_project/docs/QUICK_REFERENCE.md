# Quick Reference: Using the Generated Datasets

## Files Generated

| File | Rows | Purpose |
|------|------|---------|
| `synthetic_patient_data.csv` | 500 | Ground truth patient physiology + valve growth |
| `synthetic_scaffold_data.csv` | 500 | Scaffold parameters + simulated valve growth |
| `optimal_scaffold_designs.csv` | 500 | Best scaffold per patient + performance metrics |
| `synthetic_training_merged.csv` | 500 | Combined patient + optimal design (ready for ML) |

---

## Dataset Structures

### 1. `synthetic_patient_data.csv`
**Use Case**: Train model to predict valve growth from patient features

```
Patient_ID | Sex | Age_yrs | Height_cm | Weight_kg | BMI | BSA_m2 | Valve_t0 | Valve_t12
0          | 0   | 15.26   | 170.45    | 52.72     | 18.1| 1.58   | 21.04    | 27.41
1          | 0   | 13.99   | 157.55    | 49.72     | 20.0| 1.48   | 20.78    | 27.38
```

**Columns**:
- `Patient_ID`: 0-499
- `Sex`: 0=Female, 1=Male
- `Age_yrs`: Age at baseline (1-16 years)
- `Height_cm`, `Weight_kg`: Anthropometry
- `BMI`, `BSA_m2`: Computed metrics
-- `Valve_t0`, `Valve_t12`: Valve diameter (mm) at baseline and 12 months

**Common Use**:
```python
import pandas as pd
from sklearn.neural_network import MLPRegressor

df = pd.read_csv('synthetic_patient_data.csv')

# Feature: patient physiology
X = df[['Sex', 'Age_yrs', 'Height_cm', 'Weight_kg', 'BMI', 'BSA_m2']]

# Target: valve at baseline and 12 months
y = df[['Valve_t0', 'Valve_t12']]

# Train model
mlp = MLPRegressor(hidden_layer_sizes=(64, 64))
mlp.fit(X, y)
```

---

### 2. `synthetic_scaffold_data.csv`
**Use Case**: Understand scaffold-valve mechanics relationship

```
Patient_ID | Suture_Count | Material_Stiffness | Dissolution_Rate | Fold_Angle | SimValve_t0 | SimValve_t12
0          | 8            | 0.92               | 0.073            | 45.2       | 21.04       | 23.41
1          | 6            | 1.45               | 0.051            | 52.1       | 20.78       | 22.79
```

**Columns**:
- `Patient_ID`: Matches patient dataset
- `Suture_Count`: Number of sutures (4-10)
- `Material_Stiffness`: Material stiffness (0.5-2.5)
- `Dissolution_Rate`: Scaffold degradation rate (0.05-0.3 /year)
- `Fold_Angle`: Initial fold angle (15-60°)
- `SimValve_t*`: Simulated valve diameter at each timepoint

**Common Use**:
```python
# Train scaffold behavior model
X = df[['Suture_Count', 'Material_Stiffness', 'Dissolution_Rate', 'Fold_Angle']]
y = df[['SimValve_t0', 'SimValve_t12']]
mlp = MLPRegressor().fit(X, y)
```

---

### 3. `optimal_scaffold_designs.csv`
**Use Case**: Analyze inverse design results

```
Patient_ID | Best_Suture | Best_Stiffness | Best_Dissolution | Best_Fold | MAE_mm | Valve_t12_Target | Within_10pct
0          | 4           | 0.8            | 0.2              | 60        | 2.79   | 27.41            | False
1          | 4           | 0.8            | 0.2              | 60        | 2.91   | 27.38            | False
2          | 4           | 0.8            | 0.2              | 60        | 2.71   | 27.30            | True
```

**Columns**:
- `Patient_ID`: 0-499
- `Best_Suture` through `Best_Fold`: Optimal scaffold parameters
- `MAE_mm`: Trajectory error (best case)
- `Valve_t12_Target`: Target valve diameter at month 12
- `Within_10pct`: Boolean, True if MAE < 0.1 × Valve_t12_Target

**Common Use**:
```python
# Check success rate
success_rate = df['Within_10pct'].mean() * 100
print(f"Success rate: {success_rate}%")  # 2.8%

# Analyze optimal parameters
print(df['Best_Suture'].value_counts())     # What sutures appear most?
print(df['Best_Stiffness'].unique())        # Which stiffnesses?
print(df['MAE_mm'].describe())              # Error statistics
```

---

### 4. `synthetic_training_merged.csv`
**Use Case**: Train end-to-end model (patient features → optimal scaffold)

```
Patient_ID | Sex | Age_yrs | ... | Valve_t12 | Best_Suture | Best_Stiffness | Best_Dissolution | Best_Fold | MAE_mm | ...
0          | 0   | 15.26   | ... | 27.41     | 4           | 0.8            | 0.2              | 60        | 2.79   | ...
1          | 0   | 13.99   | ... | 27.38     | 4           | 0.8            | 0.2              | 60        | 2.91   | ...
```

**Columns**: All from `synthetic_patient_data.csv` + design columns from `optimal_scaffold_designs.csv`

**Common Use**:
```python
# Predict optimal scaffold from patient features
X = df[['Sex', 'Age_yrs', 'Height_cm', 'Weight_kg', 'BMI', 'BSA_m2']]
y = df[['Best_Suture', 'Best_Stiffness', 'Best_Dissolution', 'Best_Fold']]

mlp = MLPRegressor(hidden_layer_sizes=(128, 128))
mlp.fit(X, y)

# Now you can predict optimal scaffold for a new patient!
new_patient = [[1, 10.0, 160, 45, 17.6, 1.45]]  # [Sex, Age, Height, Weight, BMI, BSA]
optimal_params = mlp.predict(new_patient)
print(f"For this patient: Suture={optimal_params[0]:.0f}, Stiffness={optimal_params[1]:.2f}, ...")
```

---

## Quick Analyses

### Analysis 1: Valve Growth Patterns
```python
import matplotlib.pyplot as plt

 df = pd.read_csv('synthetic_patient_data.csv')

# Separate by sex
males = df[df['Sex'] == 1]
females = df[df['Sex'] == 0]

# Plot average growth trajectory
fig, ax = plt.subplots()
ax.plot([0, 12], males[['Valve_t0', 'Valve_t12']].mean(), 'b-o', label='Males')
ax.plot([0, 12], females[['Valve_t0', 'Valve_t12']].mean(), 'r-o', label='Females')
ax.set_xlabel('Time (months)')
ax.set_ylabel('Valve Diameter (mm)')
ax.legend()
plt.show()
```

---

### Analysis 2: Scaffold Parameter Convergence
```python
df = pd.read_csv('optimal_scaffold_designs.csv')

print("Optimal parameter distribution:")
print(f"  Suture count: {df['Best_Suture'].value_counts().to_dict()}")
print(f"  Stiffness: min={df['Best_Stiffness'].min()}, max={df['Best_Stiffness'].max()}")
print(f"  Dissolution: min={df['Best_Dissolution'].min()}, max={df['Best_Dissolution'].max()}")
print(f"  Fold angle: min={df['Best_Fold'].min()}, max={df['Best_Fold'].max()}")
```

---

### Analysis 3: Error vs Patient Age
```python
merged = pd.read_csv('synthetic_training_merged.csv')

# Bin by age
merged['Age_Bin'] = pd.cut(merged['Age_yrs'], bins=[0, 5, 10, 15, 16])

# Compute mean error per age bin
error_by_age = merged.groupby('Age_Bin')['MAE_mm'].agg(['mean', 'std', 'count'])
print(error_by_age)
```

---

### Analysis 4: Success Rate by Sex
```python
df = pd.read_csv('optimal_scaffold_designs.csv')
merged = pd.read_csv('synthetic_training_merged.csv')

# Add sex info
df = df.merge(merged[['Patient_ID', 'Sex']], on='Patient_ID')

print(f"Success rate (Male): {df[df['Sex']==1]['Within_10pct'].mean()*100:.1f}%")
print(f"Success rate (Female): {df[df['Sex']==0]['Within_10pct'].mean()*100:.1f}%")
```

---

## Reproducing the Dataset

To regenerate the exact same datasets:

```python
from valve_project.utils.synthetic_dataset_generator import generate_and_export_pipeline

# Generate with same random seed
results = generate_and_export_pipeline(
   n_patients=500,
   seed=42,  # ← Use same seed for reproducibility
   output_dir="./",
   verbose=True
)
```

To generate a different cohort (different random seed):

```python
# New random seed → different patients
results = generate_and_evaluate_full_pipeline(
    n_patients=500,
    seed=123,  # ← Different seed
    output_dir="./",
    verbose=True
)
```

---

## Data Quality Notes

✅ **What's Good**:
- 500 patient records (reasonable cohort size)
- Realistic anthropometry (matched to pediatric growth curves)
- Mechanistic scaffold model (not random noise)
- 4 longitudinal timepoints per patient
- Clean, well-formatted CSV output

⚠️ **Limitations**:
- Synthetic data (not real clinical data)
- Linear valve growth model (real growth may be non-linear)
- Simplified scaffold mechanics (real COMSOL is more complex)
- Random scaffold parameters (real designs are optimized)
- ±10% success rate indicates gap between theory and optimization

---

## Next Steps

1. **Use for ML Development**:
   - Train scaffold selection models
   - Predict growth trajectories
   - Patient-specific optimization

2. **Validate Assumptions**:
   - Compare growth rates with clinical literature
   - Validate scaffold mechanics with COMSOL
   - Refine model parameters based on real data

3. **Extend Pipeline**:
   - Add uncertainty quantification
   - Implement Bayesian optimization
   - Multi-objective design (MAE vs cost vs manufacturability)

---

**Generated**: February 7, 2026
**Dataset**: 500 synthetic patients, 4 timepoints, 108 design configurations per patient
**Status**: ✅ Ready for ML training and analysis
