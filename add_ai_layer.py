import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv("dhisi_structured.csv")
print("Loaded dataset:", df.shape)

X = df[['population_per_phc']]

kmeans = KMeans(n_clusters=3, random_state=42)
df['ai_cluster'] = kmeans.fit_predict(X)

cluster_means = (
    df.groupby('ai_cluster')['population_per_phc']
    .mean()
    .sort_values()
)

risk_map = {
    cluster_means.index[0]: 'Low',
    cluster_means.index[1]: 'Moderate',
    cluster_means.index[2]: 'Critical'
}

df['ai_risk_level'] = df['ai_cluster'].map(risk_map)

print("AI clustering completed")


df.to_csv("dhisi_ai_enhanced.csv", index=False)

print("AI-enhanced dataset created: dhisi_ai_enhanced.csv")
print(df[['state', 'district', 'population_per_phc', 'ai_risk_level']].head())
