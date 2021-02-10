import requests
import boto3 # Required to download assets hosted on S3
import os
from urllib.parse import urlparse
import arrow
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import json
import logging
logging.basicConfig(filename='download_data.log')

with open('secrets.json') as fa:
    jsondata = json.load(fa)

# copy your API key from dashboard.mlhub.earth and paste it in the following
API_KEY = jsondata['APIKEYMLHUB']
API_BASE = 'https://api.radiant.earth/mlhub/v1'

# COLLECTION_ID = 'ref_landcovernet_v1_labels'
COLLECTION_ID = "ref_african_crops_kenya_01_labels"
OUTPUT_FOLDER = f'data/{COLLECTION_ID}'
if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)

datasets = [
    # "ref_african_crops_kenya_01_labels",
    # "ref_african_crops_kenya_01_source",
    # "ref_african_crops_tanzania_01_labels",
    # "ref_african_crops_tanzania_01_source",
    "ref_african_crops_uganda_01_labels",
    # "ref_african_crops_uganda_01_source",
    "ref_african_crops_kenya_02_labels",
    # "ref_african_crops_kenya_02_source",
    "su_african_crops_south_sudan_labels",
    # "su_african_crops_south_sudan_source_planet",
    # "su_african_crops_south_sudan_source_s1",
    # "su_african_crops_south_sudan_source_s2",
    "su_african_crops_ghana_labels",
    # "su_african_crops_ghana_source_planet",
    # "su_african_crops_ghana_source_s1",
    # "su_african_crops_ghana_source_s2"
]

s3 = boto3.client('s3')

def download_s3(uri, path):
    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path[1:]
    s3.download_file(bucket, key, os.path.join(path, key.split('/')[-1]))
    
def download_http(uri, path):
    parsed = urlparse(uri)
    r = requests.get(uri)
    f = open(os.path.join(path, parsed.path.split('/')[-1]), 'wb')
    for chunk in r.iter_content(chunk_size=512 * 1024): 
        if chunk:
            f.write(chunk)
    f.close()

def get_download_uri(uri):
    uri = ''.join(uri)
    r = requests.get(uri, allow_redirects=False)
    return r.headers['Location']

def download(d):
    try:
        href = d[0]
        path = d[1]
    except Exception:
        href = d[0][0]
        path = d[0][1]
    try:
        download_uri = get_download_uri(href)
    except Exception:
        pass
    else:
        parsed = urlparse(download_uri)
        
        if parsed.scheme in ['s3']:
            print("S3")
            download_s3(download_uri, path)
        elif parsed.scheme in ['http', 'https']:
            # print("HTTPS/HTTP")
            try:
                download_http(download_uri, path)
            except Exception as e:
                print(e)
        
def get_source_item_assets(args):
    path = args[0]
    href = args[1]
    asset_downloads = []
    try:
        r = requests.get(href, params={'key': API_KEY})
    except:
        print('ERROR: Could Not Load', href)
        return []
    dt = arrow.get(r.json()['properties']['datetime']).format('YYYY_MM_DD')
    asset_path = os.path.join(path, dt)
    if not os.path.exists(asset_path):
        os.makedirs(asset_path)

    for _, asset in r.json()['assets'].items():
        asset_downloads.append((asset['href'], asset_path))
        
    return asset_downloads

def download_source_and_labels(item):
    labels = item.get('assets').get('labels')
    links = item.get('links')
    
    # Make the directory to download the files to
    # path = f'landcovernet/{item["id"]}/'
    global OUTPUT_FOLDER
    path = f'{OUTPUT_FOLDER}/{item["id"]}/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    source_items = []
    
    # Download the source imagery
    for link in links:
        if link['rel'] != 'source':
            continue
        source_items.append((path, link['href']))
        
    results = p.map(get_source_item_assets, source_items)
    # results.append([(labels['href'], path)])
    try:
        results.append([(labels['href'], path)])
    except TypeError:
        results.append([(path)])
            
    return results

def get_items(uri, classes=None, max_items_downloaded=None, items_downloaded=0, downloads=[]):
    print('Loading', uri, '...')
    r = requests.get(uri, params={'key': API_KEY})
    collection = r.json()
    for feature in collection.get('features', []):
        # Check if the item has one of the label classes we're interested in
        matches_class = True
        if classes is not None:
            matches_class = False
            for label_class in feature['properties'].get('labels', []):
                if label_class in classes:
                    matches_class = True
                    break
            
        # If the item does not match all of the criteria we specify, skip it
        if not matches_class:
            continue
        
        print('Getting Source Imagery Assets for', feature['id'])
        # Download the label and source imagery for the item
        
        downloads.extend(download_source_and_labels(feature))

        # Stop downloaded items if we reached the maximum we specify
        items_downloaded += 1
        if max_items_downloaded is not None and items_downloaded >= max_items_downloaded:
            return downloads
        
    # Get the next page if results, if available
    for link in collection.get('links', []):
        if link['rel'] == 'next' and link['href'] is not None:
            get_items(link['href'], classes=classes, max_items_downloaded=max_items_downloaded, items_downloaded=items_downloaded, downloads=downloads)
    
    return downloads

p = ThreadPool(20)

# n = 10
# to_download = get_items(f'{API_BASE}/collections/{COLLECTION_ID}/items',
#                         max_items_downloaded=n, downloads=[])
# for d in tqdm(to_download):
#     p.map(download, d)
#     download(d)


# # to_download = get_items(f'{API_BASE}/collections/{COLLECTION_ID}/items',
# #                         classes=['Woody Vegetation'],
# #                         max_items_downloaded=10,
# #                         downloads=[])
# # for d in tqdm(to_download):
# #     p.map(download, d)


# to_download = get_items(f'{API_BASE}/collections/{COLLECTION_ID}/items', downloads=[])
# print('Downloading Assets')
# for d in tqdm(to_download):
#     p.map(download, d)


if len(datasets)>0:
    for count,COLLECTION_ID in enumerate(datasets,1):

        OUTPUT_FOLDER = f'data/{COLLECTION_ID}'
        if not os.path.exists(OUTPUT_FOLDER):
            os.mkdir(OUTPUT_FOLDER)

        to_download = get_items(f'{API_BASE}/collections/{COLLECTION_ID}/items', downloads=[])
        
        for d in tqdm(to_download):
            p.map(download, d)

        print(f'Dataset {count} / {len(datasets)} downloaded = {COLLECTION_ID}')



