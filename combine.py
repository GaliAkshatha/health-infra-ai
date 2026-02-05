import pandas as pd
from sklearn.preprocessing import MinMaxScaler

pop_df = pd.read_excel("2011-IndiaStateDist-0000.xlsx")
phc_df = pd.read_csv("phc_infrastructure_2013.csv")

print("Files loaded")

pop_df.columns = pop_df.columns.str.lower().str.strip()

pop_df = pop_df.rename(columns={
    'state': 'state_code',          
    'district name': 'district',
    'tot_p': 'population',
    'tru': 'tru'
})

pop_df = pop_df[
    (pop_df['tru'] == 'Total') &
    (pop_df['district'].notna())
]

pop_df['state_code'] = pop_df['state_code'].astype(int)
pop_df['district'] = pop_df['district'].astype(str).str.lower().str.strip()


state_code_map = {
    1: 'jammu and kashmir',
    2: 'himachal pradesh',
    3: 'punjab',
    4: 'chandigarh',
    5: 'uttarakhand',
    6: 'haryana',
    7: 'delhi',
    8: 'rajasthan',
    9: 'uttar pradesh',
    10: 'bihar',
    11: 'sikkim',
    12: 'arunachal pradesh',
    13: 'nagaland',
    14: 'manipur',
    15: 'mizoram',
    16: 'tripura',
    17: 'meghalaya',
    18: 'assam',
    19: 'west bengal',
    20: 'jharkhand',
    21: 'odisha',
    22: 'chhattisgarh',
    23: 'madhya pradesh',
    24: 'gujarat',
    25: 'daman and diu',
    26: 'dadra and nagar haveli',
    27: 'maharashtra',
    28: 'andhra pradesh',
    29: 'karnataka',
    30: 'goa',
    31: 'lakshadweep',
    32: 'kerala',
    33: 'tamil nadu',
    34: 'puducherry',
    35: 'andaman and nicobar islands'
}

pop_df['state'] = pop_df['state_code'].map(state_code_map)

pop_df = pop_df.dropna(subset=['state'])

pop_df = pop_df[['state', 'district', 'population']]

print("Population data cleaned and state codes mapped")

phc_df.columns = phc_df.columns.str.lower().str.strip()

phc_df = phc_df.rename(columns={
    'state_ut': 'state',
    'phcs_functioning': 'phcs'
})

phc_df['state'] = phc_df['state'].astype(str).str.lower().str.strip()

phc_df = phc_df[phc_df['state'] != 'all india']

phc_df.loc[phc_df['state'] == 'telangana', 'state'] = 'andhra pradesh'

phc_df = phc_df[['state', 'phcs']]

print("PHC data cleaned")

merged_df = pop_df.merge(phc_df, on='state', how='inner')

print("Data merged — rows:", len(merged_df))

merged_df = merged_df.dropna(subset=['phcs'])
merged_df = merged_df[merged_df['phcs'] > 0]


if merged_df.empty:
    raise ValueError("Merge failed even after state-code mapping")


merged_df['population_per_phc'] = merged_df['population'] / merged_df['phcs']

scaler = MinMaxScaler(feature_range=(0, 100))
merged_df['stress_score'] = scaler.fit_transform(
    merged_df[['population_per_phc']]
)

def stress_category(score):
    if score >= 66:
        return "High Stress"
    elif score >= 33:
        return "Medium Stress"
    else:
        return "Low Stress"

merged_df['stress_category'] = merged_df['stress_score'].apply(stress_category)

final_df = merged_df[
    ['state', 'district', 'population', 'phcs',
     'population_per_phc', 'stress_score', 'stress_category']
]

final_df.to_csv("dhisi_structured.csv", index=False)

print("FINAL FILE CREATED: dhisi_structured.csv")
print(final_df.head())
