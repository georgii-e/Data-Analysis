import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def str_to_float(data_frame, column_name):
    data_frame[column_name] = abs(data_frame[column_name].apply(
        lambda x: x.replace(',', '.') if isinstance(x, str) else np.NaN).astype('float'))


def are_correlating(dataset_1, dataset_2):
    print(np.corrcoef(dataset_1, dataset_2))
    return abs(np.corrcoef(dataset_1, dataset_2)[0, 1]) > 0.8


initial_df = pd.read_csv(r'F:\Egor\Уроки\Аналіз даних\Лаб3\Data2.csv', sep=';', encoding='cp1252')
df = initial_df.copy()
df.rename(columns={'Populatiion': 'Population'}, inplace=True)
str_to_float(df, 'GDP per capita')
str_to_float(df, 'CO2 emission')
str_to_float(df, 'Area')
print(df.head(5))
df.fillna(df.mean(numeric_only=True), inplace=True)
df['Population density'] = df['Population'] / df['Area']
df.info()

features = df[['GDP per capita', 'Population density']].values

wss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=11, n_init='auto')
    kmeans.fit(features)
    wss.append(kmeans.inertia_)

plt.plot(range(1, 11), wss)
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster sum of squares')
plt.title('Elbow curve for KMeans clustering')
plt.show()

kmeans = KMeans(n_clusters=4, n_init='auto', random_state=11)
labels = kmeans.fit_predict(features)

plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis')
plt.xlabel('GDP per capita')
plt.ylabel('Population density')
plt.title('Clusters')
plt.ylim(0, 20000)
plt.show()

df['Cluster number'] = labels

fig, axs = plt.subplots(1, 4, figsize=(10, 8), sharey='all', sharex='all')
df.loc[df['Cluster number'] == 0, 'Region'].value_counts().sort_index().plot(kind='bar', ax=axs[0], title='Cluster 1')
df.loc[df['Cluster number'] == 1, 'Region'].value_counts().sort_index().plot(kind='bar', ax=axs[1], title='Cluster 2')
df.loc[df['Cluster number'] == 2, 'Region'].value_counts().sort_index().plot(kind='bar', ax=axs[2], title='Cluster 3')
df.loc[df['Cluster number'] == 3, 'Region'].value_counts().sort_index().plot(kind='bar', ax=axs[3], title='Cluster 4')
fig.suptitle('Regions by Cluster')
plt.show()

df['Region'].value_counts().plot(kind='bar', grid=True, title='Total count')
plt.show()

fig, axes = plt.subplots(1, 5)
labels = df.columns[2:7]
for i in range(len(labels)):
    ax_i = (0, i)
    axes[i].set_title(labels[i])
    axes[i].grid('-')
    axes[i].hist(df[labels[i]])
plt.show()

x = np.random.randint(0, 50, 1000)
y = x + np.random.randint(0, 20, 1000)
print(are_correlating(x, y))
y = np.array([x for x in range(1000)])
print(are_correlating(x, y))
