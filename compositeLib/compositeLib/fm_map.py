import folium
import geopandas as gpd
tiles=" http://mt1.google.com/vt/lyrs=y&z={z}&x={x}&y={y}"
attr="Google"


class Fmap():

	def __init__(self,color='blue'):
                print("creating Fmap class")
                self.color = color

	def sat_geojson(self, geojson_bb_file):
                
                my_geo = gpd.read_file(geojson_bb_file)
                center=my_geo.geometry[0].bounds[1], my_geo.geometry[0].bounds[0]
                sw = center
                ne = my_geo.geometry[0].bounds[3], my_geo.geometry[0].bounds[2]
                map3 = folium.Map(location=center, zoom_start=12,center=center,tiles=tiles,attr=attr)
                map3.fit_bounds([sw,ne])
                my_geo_json = my_geo.to_json()
                style_function= lambda x:{'color':self.color}
                folium.GeoJson(my_geo_json,style_function).add_to(map3)
                map3.add_child(folium.features.LatLngPopup())
                #map3
                return(map3)

	def map_geojson(self, geojson_bb_file):
                my_geo = gpd.read_file(geojson_bb_file)
                center=my_geo.geometry[0].bounds[1], my_geo.geometry[0].bounds[0]
                sw = center
                ne = my_geo.geometry[0].bounds[3], my_geo.geometry[0].bounds[2]
                map3 = folium.Map(location=center, zoom_start=12,center=center)
                map3.fit_bounds([sw,ne])
                my_geo_json = my_geo.to_json()
                style_function= lambda x:{'color':self.color}
                folium.GeoJson(my_geo_json,style_function).add_to(map3)
                map3.add_child(folium.features.LatLngPopup())
                #map3
                return(map3)

	def add_geojson(self, geojson_bb_file, map3, color='pink'):
                my_geo = gpd.read_file(geojson_bb_file)
                center=my_geo.geometry[0].bounds[1], my_geo.geometry[0].bounds[0]
                sw = center
                ne = my_geo.geometry[0].bounds[3], my_geo.geometry[0].bounds[2]
                #map3 = folium.Map(location=center, zoom_start=12,center=center)
                #map3.fit_bounds([sw,ne])
                my_geo_json = my_geo.to_json()
                style_function= lambda x:{'color':color}
                folium.GeoJson(my_geo_json,style_function).add_to(map3)
                return(map3)

	def ne_geojson(self, geojson_bb_file):
                my_geo = gpd.read_file(geojson_bb_file)
                center=my_geo.geometry[0].bounds[1], my_geo.geometry[0].bounds[0]
                sw = center
                ne = my_geo.geometry[0].bounds[3], my_geo.geometry[0].bounds[2]
                return(ne)
