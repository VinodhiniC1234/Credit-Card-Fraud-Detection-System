import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    # Scale only 'Amount' column
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    
    # Drop Time column (not useful)
    df = df.drop(['Time'], axis=1)
    
    return df