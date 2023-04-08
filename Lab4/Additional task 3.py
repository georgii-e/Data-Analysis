import geopandas as gpd
import pandas as pd
import geoviews as gv
import numpy as np

pd.set_option('display.precision', 3)


def str_to_float(data_frame, column_name):
    data_frame[column_name] = abs(data_frame[column_name].apply(
        lambda x: x.replace(',', '.') if isinstance(x, str) else np.NaN).astype('float'))


gv.extension('bokeh')

geometries = gpd.read_file(r'F:\Egor\Уроки\Аналіз даних\Лаб4\UKR_ADM1.shp')
ukr_dpp_df = pd.read_csv(r'F:\Egor\Уроки\Аналіз даних\Лаб4\ukr_DPP.csv', sep=';', encoding='windows-1251', header=1)
ukr_gdp_df = pd.read_csv(r'F:\Egor\Уроки\Аналіз даних\Лаб4\ukr_GDP.csv', sep=';', encoding='windows-1251', header=1)

for year in range(2006, 2017):
    str_to_float(ukr_dpp_df, str(year))

dpp_gdf = gpd.GeoDataFrame(pd.merge(geometries, ukr_dpp_df))
gdp_gdf = gpd.GeoDataFrame(pd.merge(geometries, ukr_gdp_df))

print(dpp_gdf.head())

gv.Polygons(dpp_gdf, vdims=['UKRname', '2016'], label='Population income per capita (2016)').opts(
    tools=['hover'], width=1000, height=700, color='2016',
    colorbar=True, toolbar='above', xaxis=None, yaxis=None)

gv.Polygons(gdp_gdf, vdims=['UKRname', '2016'], label='Gross regional product (2016)').opts(
    tools=['hover'], width=1000, height=700, color='2016',
    colorbar=True, toolbar='above', xaxis=None, yaxis=None)

years = [str(year) for year in range(2006, 2017)]
gdp_arr = ukr_gdp_df[years]
dpp_arr = ukr_dpp_df[years]

region_number = len(gdp_arr)

corr_coefs = []

for region_index in range(0, region_number):
    region_gdp = gdp_arr.loc[region_index, ~np.isnan(gdp_arr.iloc[region_index])]
    region_dpp = dpp_arr.loc[region_index, ~np.isnan(gdp_arr.iloc[region_index])]

    corr_coefs.append(np.corrcoef(region_gdp, region_dpp)[0, 1])

corr_df = ukr_dpp_df.drop(years, axis=1)
corr_df['GDP-DPP Correlation'] = corr_coefs
print(corr_df.head())

corr_gdf = gpd.GeoDataFrame(pd.merge(geometries, corr_df))

gv.Polygons(corr_gdf, vdims=['UKRname', 'GDP-DPP Correlation'], label='GDP-DPP Correlation').opts(
    tools=['hover'], width=1000, height=700, color='GDP-DPP Correlation',
    colorbar=True, toolbar='above', xaxis=None, yaxis=None)
