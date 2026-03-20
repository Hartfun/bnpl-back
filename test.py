import pandas as pd
df = pd.read_csv('data.csv')
df.columns = df.columns.str.strip()
df = df.drop(columns=['Timestamp'], errors='ignore').drop_duplicates().reset_index(drop=True)
print('Total rows:', len(df))

text_cols = [
    "What is your overall opinion of 'Buy Now, Pay Later' (BNPL) services like Slice or LazyPay?",
    'Describe a positive experience you have had using an EMI or BNPL service. What made it good?',
    'Describe a negative experience you have had, if any. What went wrong?',
    'What is your single biggest fear or concern about using these services regularly?'
]

for col in text_cols:
    if col in df.columns:
        filled = df[col].notna() & (df[col].str.strip() != '') & (df[col].str.strip() != 'nan')
        print(f'{filled.sum():4d} filled | {col[:60]}...')
    else:
        print(f'MISSING column: {col}')
