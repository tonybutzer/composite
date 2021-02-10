'''
new python for folio maps and
geojson files
'''

# https://medium.com/@pramukta/recipe-importing-geojson-into-shapely-da1edf79f41d

import folium
import json
from shapely.geometry import shape, GeometryCollection

from .xr_geometry import return_geo_walk_coordinates

class Ldisplay():

    def __init__(self):
        print("creating Ldisplay class")
        self.epsg = 'epsg:5072'


    def map_geojson(self, geojson_bb_file):
        MAP_TO_DISPLAY = lc_geojson_map(geojson_bb_file)
        return(MAP_TO_DISPLAY)



def lc_geojson_map(geojson_bb_file):
    """ Generates a folium map with a lat-lon bounded rectangle drawn on it. Folium maps can be 
    
    ul and lr are in crs -  ie EPSG:5072 -- usard uses this. but also others
    
    Args:
        ul and lr are in AEA meters from the origin

    Returns:
        folium.Map: A map centered on the lat lon bounds. A rectangle is drawn on this map detailing the
        perimeter of the lat,lon bounds.  A zoom level is calculated such that the resulting viewport is the
        closest it can possibly get to the centered bounding rectangle without clipping it. An 
        optional grid can be overlaid with primitive interpolation.  

    .. _Folium
        https://github.com/python-visualization/folium

    """

    poly = return_geo_walk_coordinates(geojson_bb_file)
    # print(poly)
    coords = poly[0]
    # print("MAP",coords)

    lat_lon_list = _coords_to_lat_lon_list(coords)

    centroid =  _centroid_from_geojson(geojson_bb_file)

    center = centroid[1], centroid[0]  # swicth to now less common lat, lon

    color='red'
    opacity=0.8

    polyline = lat_lon_list
    
    map_hybrid = _folium_map(center, polyline, color, opacity)

    return map_hybrid


def _folium_map(center, polyline, color, opacity):

    map_hybrid = folium.Map(
        location=center,
        # zoom_start=zoom_level,
        tiles=" http://mt1.google.com/vt/lyrs=y&z={z}&x={x}&y={y}",
        attr="Google"
    )


    map_hybrid.add_child(
        folium.features.PolyLine(
            #locations=coords,
            locations=polyline,
            color=color,
            opacity=opacity)
    )

    map_hybrid.add_child(folium.features.LatLngPopup())

    return map_hybrid


def _folium_map_add_poly(m, center, polyline, color, opacity):

    map_hybrid = m

    map_hybrid.add_child(
        folium.features.PolyLine(
            #locations=coords,
            locations=polyline,
            color=color,
            opacity=opacity)
    )

    return map_hybrid


def _coords_to_lat_lon_list(coords):

    lat_lon_list = []
    for c in coords:
        # print(c)
        l = c[1],c[0]
        # print(l)
        lat_lon_list.append(l)
    return lat_lon_list


########################################



def _centroid_from_geojson(geojson):
    ''' returns lon, lat '''

    ''' '''

    with open(geojson) as f:
        features = json.load(f)["features"]

    
    # NOTE: buffer(0) is a trick for fixing scenarios where polygons have overlapping coordinates
    geometry = GeometryCollection([shape(feature["geometry"]).buffer(0) for feature in features])

    my_plot_centroid = (geometry.centroid.xy[0][0], geometry.centroid.xy[1][0])

    return (my_plot_centroid)

