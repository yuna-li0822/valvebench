import sys
import pandas as pd
from valvebench.datasets.splits import create_group_splits, save_splits

if __name__ == '__main__':
    df = pd.read_csv('aipv/heart_valve_project/data/synthetic/synthetic_patient_data_v0.1.csv')
    train, test = create_group_splits(df, group_col='Patient_ID', test_size=0.2, random_state=42)
    print('Train size:', len(train), 'Test size:', len(test))
    save_splits(train, test, 'aipv/heart_valve_project/data/synthetic/splits')
    print('Saved train/test splits to data/synthetic/splits')
