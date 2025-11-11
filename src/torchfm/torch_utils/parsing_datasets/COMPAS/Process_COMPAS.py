import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    df1 = pd.read_csv('./train_org.csv')
    df2 = pd.read_csv('./test_org.csv')
    df3 = pd.read_csv('./validation_org.csv')
    df_union = pd.concat([df1, df2, df3]).reset_index(drop=True)
    
    #qunatize the columns that have numerical values
    n_bins = 5
    df_union['c_jail_in'] = pd.cut(df_union['c_jail_in'], bins=n_bins, labels=False)
    df_union['c_jail_out'] = pd.cut(df_union['c_jail_out'], bins=n_bins, labels=False)
    df_union['c_offense_date'] = pd.cut(df_union['c_offense_date'], bins=n_bins, labels=False)
    df_union['screening_date'] = pd.cut(df_union['screening_date'], bins=n_bins, labels=False)
    df_union['in_custody'] = pd.cut(df_union['in_custody'], bins=n_bins, labels=False)
    df_union['out_custody'] = pd.cut(df_union['out_custody'], bins=n_bins, labels=False)
    

    #shifts feature values to avoid overlaps
    prev_col = None
    for col in df_union.columns:
        if col == 'label':
            continue
        if prev_col is None:
            prev_col = col
            df_union[col] = df_union[col] - df_union[col].min()
            continue
        df_union[col] = df_union[col] - df_union[col].min()+df_union[prev_col].max()+1
        prev_col = col

    train_df, temp_df = train_test_split(df_union, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    train_df.to_csv('train.csv', index=False)
    val_df.to_csv('validation.csv', index=False)
    test_df.to_csv('test.csv', index=False)

if __name__ == "__main__":
    main()
    
