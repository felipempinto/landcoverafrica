from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import sentinelsat
from datetime import date
from collections import OrderedDict
import geopandas as gpd
import json,os
from shapely.geometry import Polygon
import random

d1 = '20210301'
d2 = '20210310'

shp = ''
outpath = ''

sentinel = 1
cloud = 0
DOWNLOAD = True

def get_bounds(shp):
    gdf = gpd.read_file(shp)
    minx, miny, maxx, maxy = gdf['geometry'][0].bounds
    p = Polygon([[minx,maxy] , [maxx,maxy] , [maxx,miny] , [minx,miny]])
    return p.to_wkt()


with open(os.path.join(os.path.dirname(__file__),'secrets.json')) as secrets_file:
    secrets = json.load(secrets_file)

user = secrets["sentinelsat_login"]
password = secrets["sentinelsat_password"]

api = SentinelAPI(user, password)#, 'https://scihub.copernicus.eu/dhus')

footprint = geojson_to_wkt(read_geojson(shp))

query_kwargs = {
        'area':footprint,
        'platformname': f'Sentinel-{sentinel}',
        'date': (d1, d2)}

if sentinel == 1:
    query_kwargs['producttype'] = 'GRD'#"SLC"
elif sentinel == 2:
    #query_kwargs['tileid'] = tile
    query_kwargs['cloudcoverpercentage'] = (0.0,cloud)

try:
    products = api.query(**query_kwargs)
except sentinelsat.sentinel.SentinelAPIError:
    bounds = get_bounds(shp)
    query_kwargs['area'] = bounds
    products = api.query(**query_kwargs)

gdf = api.to_geodataframe(products)
print(f'Images found: {len(gdf)}')


if DOWNLOAD:
    if len(gdf)>0:
        re = api.download_all(gdf.index.to_list(),outpath)
        print(re)
    else:
        print("There is no image to download")
