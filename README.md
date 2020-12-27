
<h1>Best Land to Buy in Casablanca, Morocco</h1>
<h4>Can I find a great land to buy in my city ? </h4><br>

<p>This notebook is about getting data of lands in Casablanca city in order to determine with machine learning algorithms different types of fields, to give insights about the best lands to buy.</p>

-----    Marouane BELMALLEM (2020, September). -----


Let's get the necessary elements and modules for our project


```
import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

#!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

!pip install geocoder
import geocoder

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt

# import k-means from clustering stage
from sklearn.cluster import KMeans

!pip install folium
#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library

print('Libraries imported.')
```


```
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

```

## Getting data from Web: Web Scraping Part

Here are three functions created to get data from a web page of lands' ads in Casablanca city in Morocco

* The first is gets the raw HMTL content of a single page*


```
def get_page(page):
  url = "https://www.sarouty.ma/fr/recherche?c=1&l=35&ob=mr&page=" + str(page) + "&t=5"
  
  content = requests.get(url)
  if(content.status_code == 200):
    print("The page " + str(page) + " has been successfully retrieved")
  return content
```

* This one filter the needed data in a gotten page


```
def get_data():
  data_list = []
  for page in range(1, 20):
    content = get_page(page)
    soup = BeautifulSoup(content.text, 'html.parser')
    cardlist_items = soup.find_all(class_='card-list__item')
    
    for item in cardlist_items:
       data_list.append( [ el.text for el in ( item.find_all(['h2', 'p']) ) ] )
  return data_list



```

* We need cleaned data to work with rather than bag of HTML Tags, so this function do the job


```
def data_cleaning(data):
  for element in data:
    if len(element) > 5:
      del element[-1]

    element[0] = int(''.join(filter(str.isdigit, element[0])))
    element[4] = int(element[4].replace("\xa0", "").split()[0])
    
    slicedElement = element[2].split(",")
    element[2] = slicedElement[0]
    element.append(slicedElement[1])
    
    

    
```


```
#Applying the functions
data = get_data()
data_cleaning(data)
```


```
#Data list gotten from web using BeautifulSoup scraping tool
data[:3]
```

## Raw Data to DataFrames: Pandas are Doing the Job


```
# List of data from the scraper to DataFrame
df = pd.DataFrame(data, columns =['Total Price', 'Description', 'Neighborhood', 'Type of Land', 'Total Area', 'City'])
df.head()
```


```
# We have Total Price of a piece of land and its area
# So we need its Single Price of each m2
df['Single Price'] = df['Total Price'] / df['Total Area']
df.head()
```


```
#Slice the dataframe and get the important columns
df1 = df[['Neighborhood', 'Total Price', 'Total Area', 'Single Price']].sort_values(by = 'Neighborhood').reset_index(drop = True)


#Convert the single price values to integer
df1['Single Price'] = df1['Single Price'].astype(int)

#Filter useless and wrong data
df1 = df1[df1['Total Area'] > 100]
df1 = df1[df1['Single Price'] > 100]
df1 = df1[df1['Neighborhood'] != "indéfini"]

df1.head()
```


```
df1.describe()
```


```
#Grouping and getting Avg. Price of each group
df_avg_price = df1.groupby(by='Neighborhood', as_index=False)['Single Price'].mean().sort_values(by='Single Price').reset_index(drop = True)
df_avg_price.columns = ['Neighborhood', 'Avg. Price']

df_avg_price.head()
```


```
#Now we merge the groups of neighborhoods with their avg. with the Pricipal df 
df1 = pd.merge(df1, df_avg_price, on='Neighborhood', how='outer')
df1.head()
```

* Time to get GeoData. The coordinates of each Neighborhood


```
import random

def get_coords(neighborhood):
    # initialize your variable to None
    lat_lng_coords = None
    # loop until you get the coordinates
    while(lat_lng_coords is None):
        #random value to add to a neighborhood coordinates to plot different points for each neighborhood
        rand = random.uniform(-0.009, 0.009)

        g = geocoder.arcgis('{}, Casablanca, Morocco'.format(neighborhood))
        lat_lng_coords = map(lambda x:x+rand, g.latlng)
    return lat_lng_coords
```


```
# define a function to get coordinates
coordinates1 = [ get_coords(neighborhood) for neighborhood in df1["Neighborhood"].tolist()]
```


```
# create temporary dataframe to populate the coordinates into Latitude and Longitude
coordinates = pd.DataFrame(coordinates1, columns=['Latitude', 'Longitude'])
print(coordinates.shape)
coordinates.head()
```


```
df_merge=pd.concat([df1,coordinates],axis=1)
print(df_merge.shape)
df_merge.head()
```


```
import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure(figsize=(30,10))
bar = sns.barplot(x="Neighborhood", y="Single Price", data=df_merge)

bar.set_ylabel("Avg. Price", fontsize=24)
bar.set_xlabel("Neighborhood", fontsize=24)
bar.set_xticklabels(bar.get_xticklabels(),rotation=40)
bar.set_title("Neighborhoods with their Single Price", fontsize=34)
plt.show()
```


```
address = 'Casablanca'

geolocator = Nominatim(user_agent="project")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Connecticut are {}, {}.'.format(latitude, longitude))
```


```
# create map of CT using latitude and longitude values
map_ct = folium.Map(location=[latitude, longitude], zoom_start=9)

# add markers to map
for lat, lng, Price, Neighborhood in zip(df_merge['Latitude'], df_merge['Longitude'], df_merge['Single Price'], df_merge['Neighborhood']):
    label = '{},\nPrice: {}'.format(Neighborhood, Price)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='orange',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_ct)  
    
map_ct
```


```
df_clustering = df_merge[['Total Price', 'Total Area', 'Single Price', 'Avg. Price' ]]
df_clustering.head()

```


```
df_clustering_scaled = (df_clustering - df_clustering.mean()) / df_clustering.std()
df_clustering_scaled.head()
```

## Clustering Data to Get Insights: AI Part


```
from sklearn import metrics
from scipy.spatial.distance import cdist
```


```
#Elbow Method To determine the optimal k to use as the number of clusters 
%matplotlib inline

distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df_clustering_scaled)
    distortions.append(kmeanModel.inertia_)
    
plt.figure(figsize=(10,5))
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The elbow method showng the optimal k')
plt.plot(K, distortions, 'bx-')
plt.show()
```


```
kclusters = 5

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(df_clustering_scaled)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]
```


```
# add clustering labels
#df_merge.insert(0, 'Cluster Labels', kmeans.labels_)

ct_merged = df_merge

# Add latitude/longitude for each neighborhood
#ct_merged = ct_merged.join(df_merge.set_index('Neighborhood'), on='Neighborhood')
ct_merged.head() # check the last columns!
```


```
# create map of CT using latitude and longitude values
map_ct = folium.Map(location=[latitude, longitude], zoom_start=9)

# add markers to map
for lat, lng, Price, Neighborhood, cluster_label in zip(ct_merged['Latitude'], ct_merged['Longitude'], ct_merged['Single Price'], ct_merged['Neighborhood'], ct_merged['Cluster Labels']):
    label = '{},\nPrice: {}'.format(Neighborhood, Price)
    label = folium.Popup(label, parse_html=True)
    #Colors for different clusters
    cluster_color = {0:"#FF0000", 1:"#00FF00", 2:"#FF00FF", 3:"#0000FF", 4:"#FF00FF"}
        
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='white',
        fill=True,
        fill_color = cluster_color[cluster_label],
        fill_opacity=0.7,
        parse_html=False).add_to(map_ct)  
    
map_ct
```

*This Project has just begun, I wish I could make it better in futur*

**Thank you for reading this Notebook hoping it was helpful**

Linkedin: https://www.linkedin.com/in/marouane-belmallem

Github: https://github.com/BELMALLEM
