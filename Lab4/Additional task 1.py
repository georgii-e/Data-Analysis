import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
from scipy.spatial import distance

cities = {'City': ['Kyiv', 'Lviv', 'Odessa', 'Sumy', 'Luhansk'],
          'X coordinate': [387, 90, 395, 580, 788],
          'Y coordinate': [144, 187, 414, 110, 275],
          'Population': [2884, 721, 993, 264, 409]}
df = pd.DataFrame(cities)
map_img = img.imread(r'F:\Egor\Уроки\Аналіз даних\Лаб4\Ukraine.jpg')

fig, ax = plt.subplots(figsize=(15,15))
ax.imshow(map_img)
ax.scatter(df.loc[:, 'X coordinate'], df.loc[:, 'Y coordinate'], s=df.loc[:, 'Population'] * 3, alpha=0.5, linewidth=2)
ax.axis('off')
plt.show()

distances = distance.cdist(df.loc[:, ['X coordinate', 'Y coordinate']], df.loc[:, ['X coordinate', 'Y coordinate']], 'euclidean')
city_A, city_B = np.unravel_index(distances.argmax(), distances.shape)
pixel_distance = distances[city_A, city_B]

ukraine_width_km = 1316
km_per_pixel = ukraine_width_km / map_img.shape[1]
km_distance = pixel_distance * km_per_pixel

print(f'The longest distance is between {df.loc[city_A,"City"]} and {df.loc[city_B,"City"]}.')
print(f'Distance in pixels: {pixel_distance:.2f}')
print(f'Distance in kilometers: {km_distance:.2f}')