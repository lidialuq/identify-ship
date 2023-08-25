import pandas
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from settings import PROJECT_ROOT


csv = pandas.read_csv(os.path.join(PROJECT_ROOT, 'data', 'MarineTraffic_Vessels_Export_2023-08-18.csv'))
print(csv.columns)

# IMO is a unique identifier for ships. Small ships don't have IMO, 
# so we will remove them from the dataset (could use MMSI instead, 
# but it's not available in the dataset I downloaded)
imo = csv['Imo'].tolist()
imo = [x for x in imo if x != 0]
print('Number of ships with IMO within 50 nautical miles of Oslo: {}'.format(len(imo)))

# Make a bash script to call the scraper, that will download the first 
# 10 images appearing in the bing image search for each ship
print('Making bash script to call the image scraper')
with open('run_scraper.sh', 'w') as f:
    for i in imo:
        search_term = f"'IMO {i}'"
        cmd = (f'python ./google-images-download/bing_scraper.py --search {search_term} --limit 30 --download '
        '--chromedriver /home/lidia/CRAI-NAS/all/lidfer/identify-ship/image_scraper/chromedriver-linux64/chromedriver '
        '-o /home/lidia/CRAI-NAS/all/lidfer/identify-ship/data/images')
        f.write(cmd + '\n')



