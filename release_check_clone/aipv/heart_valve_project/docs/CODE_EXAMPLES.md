Code Examples: Using the Synthetic Valve Dataset

This document contains copy-paste-ready examples for common tasks.

---

1) Load and explore data

```python
import pandas as pd
import numpy as np

# Load patient data
patient_df = pd.read_csv('synthetic_patient_data.csv')
scaffold_df = pd.read_csv('synthetic_scaffold_data.csv')
design_df = pd.read_csv('optimal_scaffold_designs.csv')
merged_df = pd.read_csv('synthetic_training_merged.csv')

print(patient_df.shape)
```

... (examples moved from original CODE_EXAMPLES.py) ...
