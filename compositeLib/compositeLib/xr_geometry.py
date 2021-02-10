""" This is where geometry stuff lives more to come ..."""

from geojson import Polygon
from pyproj import Proj
import pygeoj


def return_aoi_polygon(aoi_ul, aoi_lr):

    ul = aoi_ul
    lr = aoi_lr
    ur = (ul[0],lr[1])
    ll = (lr[0],ul[1])

    aoi_poly = Polygon(coordinates = [ul, ur, lr, ll, ul])

    return aoi_poly


def geo_translate(lat,lon,epsg="epsg:5072"):
    """ converts lat lon to albers X and Y

        Returns: x, y
    """
    # print(lat,lon)

    p = Proj(init=epsg) # EPSG code AEA


    x,y = p(lon,lat)

    # print(x,y)
    return(x,y)


def geo_untranslate(x,y,epsg="epsg:5072"):
    """ converts albers X and Y to lat, lon

        Returns: lat, lon
    """
    # print(x,y)

    p = Proj(init=epsg) # EPSG code AEA


    glon, glat = p(x, y, inverse=True)


    # print(glat,glon)
    return(glat,glon)

def _return_geo_walk_coordinates(geo_file):
    testfile = pygeoj.load(geo_file)
    for feature in testfile:
        return (feature.geometry.coordinates)


def return_geo_walk_coordinates(geo_file):
    return(_return_geo_walk_coordinates(geo_file))


def bounding_box_tuple_from_geojson(aoi_geojson_file_name):
        coords = _return_geo_walk_coordinates(aoi_geojson_file_name)
        print(coords)
        bbox='hello'
        bbox = (coords[0][3], coords[0][1])
        return(bbox)


import osgeo.osr as osr

# def wkt2epsg(wkt, epsg='/usr/local/share/proj/epsg', forceProj4=False):
def wkt2epsg(wkt, epsg='/usr/share/proj/epsg', forceProj4=False):

    ''' Transform a WKT string to an EPSG code

    Arguments
    ---------

    wkt: WKT definition
    epsg: the proj.4 epsg file (defaults to '/usr/local/share/proj/epsg')
    forceProj4: whether to perform brute force proj4 epsg file check (last resort)

    Returns: EPSG code

    '''
    code = None
    p_in = osr.SpatialReference()
    s = p_in.ImportFromWkt(wkt)
    if s == 5:  # invalid WKT
        return None
    if p_in.IsLocal() == 1:  # this is a local definition
        return p_in.ExportToWkt()
    if p_in.IsGeographic() == 1:  # this is a geographic srs
        cstype = 'GEOGCS'
    else:  # this is a projected srs
        cstype = 'PROJCS'

    print(cstype)
    an = p_in.GetAuthorityName(cstype)
    ac = p_in.GetAuthorityCode(cstype)
    print(an, ac)
    if an is not None and ac is not None:  # return the EPSG code
        return '%s:%s' % \
            (p_in.GetAuthorityName(cstype), p_in.GetAuthorityCode(cstype))
    else:  # try brute force approach by grokking proj epsg definition file
        p_out = p_in.ExportToProj4()
        print(p_out)
        if p_out:
            if forceProj4 is True:
                return p_out
            f = open(epsg)
            for line in f:
                if line.find(p_out) != -1:
                    m = re.search('<(\\d+)>', line)
                    if m:
                        code = m.group(1)
                        break
            if code:  # match
                return 'EPSG:%s' % code
            else:  # no match
                return None
        else:
            # return None
            return 'EPSG:5072'
