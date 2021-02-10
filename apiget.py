import requests
import json

with open('secrets.json') as fa:
    jsondata = json.load(fa)

headers = {
  'Accept': 'application/json',
  'ApiKeyAuth':jsondata['APIKEYMLHUB']
}

r = requests.get(f'https://api.radiant.earth/mlhub/v1/collections', headers = headers)

json = r.json()
for i in json['collections']:
    print(i['id'])

# ref_african_crops_kenya_01_labels
# ref_african_crops_kenya_01_source
# ref_african_crops_tanzania_01_labels
# ref_african_crops_tanzania_01_source
# ref_african_crops_uganda_01_labels
# ref_african_crops_uganda_01_source
# microsoft_chesapeake_landsat_leaf_off
# microsoft_chesapeake_buildings
# sn4_AOI_6_Atlanta
# ref_african_crops_kenya_02_labels
# ref_african_crops_kenya_02_source
# microsoft_chesapeake_naip
# microsoft_chesapeake_nlcd
# microsoft_chesapeake_lc
# microsoft_chesapeake_landsat_leaf_on
# sn1_AOI_1_RIO
# sn2_AOI_2_Vegas
# sn2_AOI_5_Khartoum
# sn3_AOI_2_Vegas
# sn3_AOI_5_Khartoum
# sn5_AOI_8_Mumbai
# sn6_AOI_11_Rotterdam
# sn2_AOI_3_Paris
# sn2_AOI_4_Shanghai
# sn3_AOI_3_Paris
# sn3_AOI_4_Shanghai
# sn5_AOI_7_Moscow
# su_sar_moisture_content
# bigearthnet_v1_source
# bigearthnet_v1_labels
# ref_landcovernet_v1_source
# ref_landcovernet_v1_labels
# nasa_tropical_storm_competition_train_labels
# nasa_tropical_storm_competition_train_source
# nasa_tropical_storm_competition_test_source
# su_african_crops_south_sudan_labels
# su_african_crops_south_sudan_source_planet
# su_african_crops_south_sudan_source_s1
# su_african_crops_south_sudan_source_s2
# su_african_crops_ghana_labels
# su_african_crops_ghana_source_planet
# su_african_crops_ghana_source_s1
# su_african_crops_ghana_source_s2
# sn7_train_source
# sn7_test_source
# sn7_train_labels
# open_cities_ai_challenge_test
# open_cities_ai_challenge_train_tier_1_labels
# open_cities_ai_challenge_train_tier_1_source
# open_cities_ai_challenge_train_tier_2_labels
# open_cities_ai_challenge_train_tier_2_source