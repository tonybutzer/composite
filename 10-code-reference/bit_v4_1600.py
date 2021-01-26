import base64
import argparse
from argparse import RawTextHelpFormatter
from functools import lru_cache, partial
import multiprocessing as mp
import datetime as dt
from collections import defaultdict
import time
from typing import Union, List, Tuple, Callable, Iterable, TypedDict, NamedTuple
import random

import numpy as np
import numpy.ma as ma
import requests
from osgeo import gdal, ogr
#import ogr
#import gdal
import tqdm


# typing for custom objects
class DataStack(TypedDict):
    """
    Container for storing chip data
    """
    ulx: float
    uly: float
    acquired: List[str]
    data: np.ndarray
    source: List[str]
    ubids: List[str]


class BandMap(NamedTuple):
    """
    Simple immutable container for mapping semantic layer names to ubids
    """
    sr_blues: Tuple[str]
    sr_greens: Tuple[str]
    sr_reds: Tuple[str]
    sr_nirs: Tuple[str]
    sr_swir1s: Tuple[str]
    sr_swir2s: Tuple[str]
    bts: Tuple[str]
    qas: Tuple[str]


class QAMap(NamedTuple):
    """
    Container to hold QA bit-value offsets
    """
    fill: int
    clear: int
    water: int
    shadow: int
    snow: int
    cloud: int
    cl_conf1: int
    cl_conf2: int
    cirrus1: int
    cirrus2: int
    occulsion: int


# Establish basic mappings between sensors and a semantic layer names.
L8BandMap = BandMap(('LC08_SRB2',), ('LC08_SRB3',), ('LC08_SRB4',), ('LC08_SRB5',),
                    ('LC08_SRB6',), ('LC08_SRB7',), ('LC08_BTB10',), ('LC08_PIXELQA',))

L7BandMap = BandMap(('LE07_SRB1',), ('LE07_SRB2',), ('LE07_SRB3',), ('LE07_SRB4',),
                    ('LE07_SRB5',), ('LE07_SRB7',), ('LE07_BTB6',), ('LE07_PIXELQA',))

L5BandMap = BandMap(('LT05_SRB1',), ('LT05_SRB2',), ('LT05_SRB3',), ('LT05_SRB4',),
                    ('LT05_SRB5',), ('LT05_SRB7',), ('LT05_BTB6',), ('LT05_PIXELQA',))

L4BandMap = BandMap(('LT04_SRB1',), ('LT04_SRB2',), ('LT04_SRB3',), ('LT04_SRB4',),
                    ('LT04_SRB5',), ('LT04_SRB7',), ('LT04_BTB6',), ('LT04_PIXELQA',))

ARDGroups = (('l4', L4BandMap), ('l5', L5BandMap), ('l7', L7BandMap), ('l8', L8BandMap))

PixelQA = QAMap(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

_cu_url = 'http://lcmap-hpc.cr.usgs.gov/ARD_CU_C01_V01'
_ak_url = 'http://lcmap-hpc.cr.usgs.gov/ARD_AK_C01_V01'
_hi_url = 'http://lcmap-hpc.cr.usgs.gov/ARD_HI_C01_V01'

#####################################################################
#                    Data requests to the LCMAP services
#####################################################################
def retry(retries):
    def retry_dec(func):
        def wrapper(*args, **kwargs):
            count = 1
            while True:
                try:
                    return func(*args, **kwargs)
                except:
                    count += 1
                    if count > retries:
                        raise
                    time.sleep(random.randint(5,60))
        return wrapper
    return retry_dec


@retry(20)
def getjson(resource: str, params: dict = None) -> Union[dict, list]:
    """
    Performs a GET on the resource with the given params.
    Assumes the response is a JSON, and tries to convert as such.
    """
    resp = requests.get(resource, params=params, headers={'Connection': 'close'}, timeout=10)

    if not resp.ok:
        resp.raise_for_status()

    return resp.json()


@lru_cache()
def regionresource(region: str) -> str:
    """
    Return the resource url for the given region.
    """
    if region == 'cu':
        return _cu_url
    if region == 'ak':
        return _ak_url
    if region == 'hi':
        return _hi_url

    raise ValueError


def getchips(x: float, y: float, acquired: str, ubid: str, resource: str) -> List[dict]:
    """
    Make a request to the HTTP API for some chip data.
    """
    chip_resource = f'{resource}/chips'
    params = {'x': x,
              'y': y,
              'acquired': acquired,
              'ubid': ubid}

    return getjson(chip_resource, params)


@lru_cache()
def getregistry(resource: str) -> List[dict]:
    """
    Retrieve the spec registry from the API.
    """
    resource = f'{resource}/registry'
    return getjson(resource)


@lru_cache()
def getgrid(resource: str) -> List[dict]:
    """
    Retrieve the tile and chip definitions for the grid (geospatial transformation information)
    from the API.
    """
    resource = f'{resource}/grid'
    return getjson(resource)


@lru_cache()
def getsnap(x: float, y: float, resource: str) -> dict:
    """
    Get the containing chip and tile coordinate information
    """
    resource = f'{resource}/grid/snap'

    return getjson(resource, params={'x': x, 'y': y})


@lru_cache()
def getspec(ubid: str, resource: str) -> dict:
    """
    Retrieve the appropriate spec information for the corresponding ubid.
    """
    registry = getregistry(resource)
    return next(filter(lambda x: x['ubid'] == ubid, registry), None)


def tonumpy(chip: dict, spec: dict) -> dict:
    """
    Convert the data field in a chip response to a numpy array.
    """
    out = {k: v for k, v in chip.items()}
    data = base64.b64decode(out['data'])
    out['data'] = np.frombuffer(data, spec['data_type'].lower()).reshape(*spec['data_shape'])

    return out


def getard(x: float, y: float, ranges: Iterable, ubid: str, resource: str) -> List[dict]:
    """
    Request ARD data and convert the data responses to Numpy array's.
    """
    data = []
    for acquired in ranges:
        data.extend(getchips(x, y, acquired, ubid, resource))
    spec = getspec(ubid, resource)

    return [tonumpy(chip, spec) for chip in data]


def requestgroup(x: float, y: float, ranges: Iterable, group: List[str],
                 resource: str, filter_fill: bool = True) -> List[dict]:
    """
    Request all ubids in an associated grouping.
    """
    ret = []
    for u in group:
        data = getard(x, y, ranges, u, resource)
        if data:
            if filter_fill:
                ret.extend(filterfill(data))
            else:
                ret.extend([d for d in data])

    return ret


def filterfill(chipstack: List[dict]) -> filter:
    """
    Filter out fill chips from the stack. Should use the registry service,
    but that is currently incomplete ...
    """
    ubid = chipstack[0]['ubid']
    if 'SRB' in ubid:
        return filter(lambda chip: not np.all(chip['data'].ravel() == -9999), chipstack)
    elif 'PIXELQA' in ubid:
        return filter(lambda chip: not np.all(chip['data'].ravel() == 1), chipstack)
    elif 'BTB' in ubid:
        return filter(lambda chip: not np.all(chip['data'].ravel() == -9999), chipstack)
    else:
        raise ValueError


def sensor_groups(sensors: List, sensor_groups: Tuple[Tuple[str, BandMap]] = ARDGroups) -> dict:
    """
    Allow some choice into which sensors are considered.
    """
    band_groups = defaultdict(list)

    for sensor, bandmap in sensor_groups:
        if sensor in sensors or sensors[0] == 'all':
            for bandname, ubids in bandmap._asdict().items():
                band_groups[bandname].extend(ubids)

    return band_groups


#######################################################################
###              Chips -> datastacks and datastack operations
#######################################################################
def datastack(chipstack: List[dict], cx: int, cy: int, fill_val: float = 0) -> DataStack:
    """
    Bring together the list of chips into a single numpy data stack. Try to keep provenance.
    """
    if not chipstack:
        return {'ulx': cx,
                'uly': cy,
                'acquired': ['1900-01-01'],
                'data': np.full(shape=(1, 100, 100), fill_value=fill_val),
                'source': ['empty'],
                'ubids': ['empty']}

    out = {'ulx': chipstack[0]['x'],
           'uly': chipstack[0]['y'],
           'acquired': [],
           'data': np.zeros(shape=(len(chipstack), 100, 100), dtype=chipstack[0]['data'].dtype),
           'source': [],
           'ubids': []}

    for didx, idx in enumerate(np.argsort([c['acquired'] for c in chipstack])):
        out['acquired'].append(chipstack[idx]['acquired'])
        out['data'][didx] = chipstack[idx]['data']
        out['source'].append(chipstack[idx]['source'])
        out['ubids'].append(chipstack[idx]['ubid'])
    out['ubids'] = set(out['ubids'])

    return out


def dstack_request(x: float, y: float, ranges: Iterable, group: List[str], resource: str,
                   filter_fill: bool = True, fill_val: float = 0) -> DataStack:
    """
    Simple wrapper function to requestgroup to convert into a datastack
    """
    return datastack(requestgroup(x, y, ranges, group, resource, filter_fill), x, y, fill_val)


def dstack_idx(idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes 2-d index returns from numpy.argmin or numpy.argmax on a stack where axis=0 and turns it into
    a tuple of tuples for indexing back into the 3-d array
    """
    return (idxs,
            np.repeat(np.arange(100).reshape(-1, 1), repeats=100, axis=1),
            np.array(list(range(100)) * 100).reshape(100, 100))


def dstack_addmask(dstack: DataStack, mask: np.ndarray) -> DataStack:
    """
    Add a mask layer to an established datastack
    """
    return {'ulx': dstack['ulx'],
            'uly': dstack['uly'],
            'acquired': dstack['acquired'],
            'data': ma.masked_array(dstack['data'], dtype=dstack['data'].dtype, mask=mask),
            'source': dstack['source'],
            'ubids': dstack['ubids']}


def dstack_median(dstack: DataStack) -> DataStack:
    return {'ulx': dstack['ulx'],
            'uly': dstack['uly'],
            'acquired': dstack['acquired'],
            'data': np.ma.median(dstack['data'], axis=0),
            'source': dstack['source'],
            'ubids': dstack['ubids']}


def dstack_date(dstack: dict) -> dict:
    """
    Create a datastack of integers representing YYYYmmdd of each observation
    """
    out = {'ulx': dstack['ulx'],
           'uly': dstack['uly'],
           'acquired': dstack['acquired'],
           'data': np.zeros(shape=dstack['data'].shape),
           'source': dstack['source'],
           'ubids': dstack['ubids']}

    for idx, acq in enumerate(dstack['acquired']):
        out['data'][idx] = iso_toint(acq)

    return out


#####################################################################
#              PixelQA stuff
#####################################################################
#####################################################################
#              Observation Masking Based on PixelQA
#####################################################################
def mask_cloud(qa_dstack: DataStack, qamap: QAMap = PixelQA) -> np.ndarray:
    """
    Return a boolean mask indicating where the observations
    are cloud according to PixelQA.
    """
    return qa_dstack['data'] & 1 << qamap.cloud > 0


def mask_hccloud(qa_dstack: DataStack, qamap: QAMap = PixelQA) -> np.ndarray:
    """
    Return a boolean mask indicating where the observations
    are high confidence cloud according to PixelQA.
    """
    return ((qa_dstack['data'] & 1 << qamap.cl_conf1 > 0) &
            (qa_dstack['data'] & 1 << qamap.cl_conf1 > 0))


def mask_mccloud(qa_dstack: DataStack, qamap: QAMap = PixelQA) -> np.ndarray:
    """
    Return a boolean mask indicating where the observations
    are medium confidence cloud according to PixelQA.
    """
    return (~(qa_dstack['data'] & 1 << qamap.cl_conf1 > 0) &
            (qa_dstack['data'] & 1 << qamap.cl_conf1 > 0))


def mask_lccloud(qa_dstack: DataStack, qamap: QAMap = PixelQA) -> np.ndarray:
    """
    Return a boolean mask indicating where the observations
    are low confidence cloud according to PixelQA.
    """
    return ((qa_dstack['data'] & 1 << qamap.cl_conf1 > 0) &
            ~(qa_dstack['data'] & 1 << qamap.cl_conf1 > 0))


def mask_snow(qa_dstack: DataStack, qamap: QAMap = PixelQA) -> np.ndarray:
    """
    Return a boolean mask indicating where the observations
    are snow according to PixelQA.
    """
    return qa_dstack['data'] & 1 << qamap.snow > 0


def mask_shadow(qa_dstack: DataStack, qamap: QAMap = PixelQA) -> np.ndarray:
    """
    Return a boolean mask indicating where the observations
    are cloud shadow according to PixelQA.
    """
    return qa_dstack['data'] & 1 << qamap.shadow > 0


def mask_clear(qa_dstack: DataStack, qamap: QAMap = PixelQA) -> np.ndarray:
    """
    Return a boolean mask indicating where the observations
    are clear according to PixelQA.
    """
    return qa_dstack['data'] & 1 << qamap.clear > 0


def mask_water(qa_dstack: DataStack, qamap: QAMap = PixelQA) -> np.ndarray:
    """
    Return a boolean mask indicating where the observations
    are water according to PixelQA.
    """
    return qa_dstack['data'] & 1 << qamap.water > 0


def mask_hccirrus(qa_dstack: DataStack, qamap: QAMap = PixelQA) -> np.ndarray:
    """
    Return a boolean mask indicating where the observations
    are high confidence cirrus according to PixelQA.
    """
    return ((qa_dstack['data'] & 1 << qamap.cirrus1 > 0) &
            (qa_dstack['data'] & 1 << qamap.cirrus2 > 0))


def mask_mccirrus(qa_dstack: DataStack, qamap: QAMap = PixelQA) -> np.ndarray:
    """
    Return a boolean mask indicating where the observations
    are medium confidence cirrus according to PixelQA.
    """
    return (~(qa_dstack['data'] & 1 << qamap.cirrus1 > 0) &
            (qa_dstack['data'] & 1 << qamap.cirrus2 > 0))


def mask_lccirrus(qa_dstack: DataStack, qamap: QAMap = PixelQA) -> np.ndarray:
    """
    Return a boolean mask indicating where the observations
    are low confidence cirrus according to PixelQA.
    """
    return ((qa_dstack['data'] & 1 << qamap.cirrus1 > 0) &
            ~(qa_dstack['data'] & 1 << qamap.cirrus2 > 0))


def mask_occlusion(qa_dstack: DataStack, qamap: QAMap = PixelQA) -> np.ndarray:
    """
    Return a boolean indicating mask where the observations
    are terrain occluded according to PixelQA.
    """
    return qa_dstack['data'] & 1 << qamap.occulsion > 0


def standard_mask(qa_dstack: DataStack, qamap: QAMap = PixelQA) -> np.ndarray:
    """
    Construct a standard mask that indicates conditions
    other than clear or water, from pixelQA.
    """
    return ~(mask_water(qa_dstack, qamap) | mask_clear(qa_dstack, qamap))

def standard_mask_snow(qa_dstack: DataStack, qamap: QAMap = PixelQA) -> np.ndarray:
    """
    Construct a standard mask that indicates conditions
    other than clear or water, from pixelQA.
    """
    return ~(mask_water(qa_dstack, qamap) | mask_clear(qa_dstack, qamap) | mask_snow(qa_dstack, qamap))



#####################################################################
#              Spatial functions
#####################################################################
def align(x: float, y: float, resource: str) -> Tuple[int, int]:
    """
    Aligns the coordinate to the chip grid.
    """
    x, y = getsnap(x, y, resource)['chip']['proj-pt']
    return int(x), int(y)


def rasterize(shapepath: str, resource: str, pixel_size: int = 30) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Adapted from:
    http://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html#convert-vector-layer-to-array
    """
    # Open the data source and read in the extent
    source_ds = ogr.Open(shapepath)
    source_layer = source_ds.GetLayer()
    # source_srs = source_layer.GetSpatialRef()
    x_min, x_max, y_min, y_max = source_layer.GetExtent()
    x_min, y_max = align(x_min, y_max, resource)
    x_max, y_min = align(x_max, y_min, resource)
    x_max += 3000  # Align will pull it in, so let's buffer it back out.
    y_min -= 3000

    # Create the destination data source
    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)
    target_ds = gdal.GetDriverByName('MEM').Create('', x_res, y_res, gdal.GDT_Byte)
    target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    band = target_ds.GetRasterBand(1)
    # band.SetNoDataValue(NoData_value)

    # Rasterize
    gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[1])

    # Read as array
    array = band.ReadAsArray()

    source_ds = None
    target_ds = None

    return array, (x_min, y_max, x_max, y_min)


def buildaffine(ulx: int, uly: int, size: int = 30) -> Tuple[int, int ,int, int, int ,int]:
    """
    Returns a standard GDAL geotransform affine tuple for 30m pixels.
    """
    return ulx, size, 0, uly, 0, -size


def transform_geo(x: float, y: float, affine: Tuple[int, int, int, int, int, int]) -> Tuple[int, int]:
    """
    Transform from a geospatial (x, y) to (row, col).
    """
    col = (x - affine[0]) / affine[1]
    row = (y - affine[3]) / affine[5]
    return int(row), int(col)


def transform_rowcol(row: int, col: int, affine: Tuple[int, int, int, int, int, int]) -> Tuple[int, int]:
    """
    Tranform from (row, col) to geospatial (x, y).
    """
    x = affine[0] + col * affine[1]
    y = affine[3] + row * affine[5]
    return x, y


def buildrequestls(trutharr: np.ndarray, ulx: int, uly: int) -> List[Tuple[int, int]]:
    """
    Build a list of what chips to actually request based on an array of 0 or other.
    Assumes a 30m pixel size and the ulx/uly have already been aligned.
    """
    aff = buildaffine(ulx, uly, 30)
    rows, cols = trutharr.shape
    truth = trutharr.astype(np.bool)

    return [transform_rowcol(row, col, aff)
            for row in range(0, rows, 100)
            for col in range(0, cols, 100)
            if np.any(truth[row:row + 100, col:col + 100])]


#####################################################################
#              Helper functions
#####################################################################
def iso_todatetime(iso_string: str) -> dt.datetime:
    """
    Turn the date string into a datetime object
    """
    return dt.datetime.strptime(iso_string[:10], '%Y-%m-%d')


def iso_toint(iso_string: str) -> int:
    """
    Convert an ordinal date into an integer representing YYYYmmdd
    """
    return int(iso_todatetime(iso_string).strftime('%Y%m%d'))


def distance_absolute(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
    Calculate the absolute distance between the values in the two different arrays
    """
    return np.abs(arr1 - arr2)


def distance_median(arr: np.ndarray) -> np.ndarray:
    """
     Calculate the absolute distance each value is from the median
    """
    median = np.ma.median(arr, axis=0)
    return distance_absolute(arr, median)


def sum_squares(arrs: List[np.ndarray]) -> np.ndarray:
    """
    Square and then sum all the values (element wise) in the given arrays
    """
    return np.ma.sum([np.ma.power(a, 2) for a in arrs], axis=0)


def distance_overall(spectral_dstacks: List[DataStack]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the per-pixel index location for observations that come closest to the overall median value
    """
    euc_dist = np.ma.sqrt(sum_squares([distance_median(d['data']) for d in spectral_dstacks]))

    return dstack_idx(np.ma.argmin(euc_dist, axis=0))


#####################################################################
#              Selection functions
#####################################################################
def dstack_check(dstacks: List[DataStack]) -> bool :
    """
    Check all the dstacks to make sure they contain the same data depth
    """
    first = set(dstacks[0]['acquired'])

    for d in dstacks:
        if set(d['acquired']) != first:
            return False

    return True


def standard_request(x: float, y: float, ranges: str, resource: str, band_groups: dict, composite: str):
    """
    Request all the associated bands, and apply the standard mask
    """
    reds = dstack_request(x, y, ranges, band_groups['sr_reds'], resource)
    greens = dstack_request(x, y, ranges, band_groups['sr_greens'], resource)
    blues = dstack_request(x, y, ranges, band_groups['sr_blues'], resource)
    nirs = dstack_request(x, y, ranges, band_groups['sr_nirs'], resource)
    swir1s = dstack_request(x, y, ranges, band_groups['sr_swir1s'], resource)
    swir2s = dstack_request(x, y, ranges, band_groups['sr_swir2s'], resource)
    qas = dstack_request(x, y, ranges, band_groups['qas'], resource, fill_val=1)

    if not dstack_check([reds, greens, blues, nirs, swir1s, swir2s, qas]):
        return None

    if composite == 'leafoff' or composite == 'leafon' or composite == 'reference':
        mask1 = standard_mask(qas)
    elif composite == 'leafoff_nosnow':
        mask = standard_mask_snow(qas)


    mask2 = [
            (
                (reds['data'] +
                 greens['data'] +
                 blues['data'] +
                 nirs['data'] +
                 swir1s['data'] +
                 swir2s['data']
                )
                < 1600)
            ]

    print (mask1)
    print (mask2)
    mask = mask1|mask2



    return (dstack_addmask(blues, mask),
            dstack_addmask(greens, mask),
            dstack_addmask(reds, mask),
            dstack_addmask(nirs, mask),
            dstack_addmask(swir1s, mask),
            dstack_addmask(swir2s, mask))


def zero_out(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Zero out the values indicated in the mask
    """
    out = np.array(data)
    out[mask] = 0

    return out


def median_value(x: float, y: float, ranges: str, resource: str, band_groups: dict, composite: str) -> dict:
    data = standard_request(x, y, ranges, resource, band_groups, composite)

    if data is None:
        return None

    indxs = distance_overall(data)
    total_masked = np.ma.sum(np.ma.getmask(data[0]['data']), axis=0)
    nodata_mask = total_masked == data[0]['data'].shape[0]

    return {'x': data[0]['ulx'],
            'y': data[0]['uly'],
            'median': [zero_out(d['data'][indxs], nodata_mask) for d in data],
            'dates': zero_out(dstack_date(data[0])['data'][indxs], nodata_mask),
            'masked': np.array(total_masked),
            'not-masked': np.array(np.ma.sum(~np.ma.getmask(data[0]['data']), axis=0))}


#####################################################################
#              Zug-zug functions
#####################################################################
def worker(coord: Tuple[float, float], ranges: Iterable, resource: str, func: Callable, band_groups: dict, composite: str):
    cx, cy = coord
    return func(cx, cy, ranges, resource, band_groups, composite)


def create_ds(path: str, extent: Tuple[int, int, int, int], datatype: int,
              resource: str, driver: str, bands: int=1) -> None:
    """
    Create the initial file to write data to
    """
    cols = int((extent[2] - extent[0]) / 30)
    rows = int((extent[1] - extent[3]) / 30)

    ds = gdal.GetDriverByName(driver).Create(path, cols, rows, bands, datatype, options=[])

    ds.SetGeoTransform((extent[0], 30, 0, extent[1], 0, -30))
    ds.SetProjection(getgrid(resource)[0]['proj'])

    return ds


def write_arr(path: str, data: np.ndarray, col_off: int, row_off: int, band: int) -> None:
    ds = gdal.Open(path, gdal.GA_Update)
    ds.GetRasterBand(band).WriteArray(data, col_off, row_off)
    ds = None


def mask_chip(truth_arr: np.ndarray, chip_data: np.ndarray, row_off: int, col_off: int):
    out_chip = np.zeros_like(chip_data)

    mask = truth_arr[row_off:row_off + 100, col_off:col_off + 100]
    out_chip[mask] = chip_data[mask]

    return out_chip


def main_median(shapepath: str, ranges: Iterable, out_file: str, driver: str, cpu: int, region: str, sensors: list, composite: list):
    """
    """
    resource = regionresource(region)
    truth, extent = rasterize(shapepath, resource)
    band_groups = sensor_groups(sensors)

    func = partial(worker,
                   ranges=ranges,
                   resource=resource,
                   func=median_value,
                   band_groups=band_groups,
                   composite=composite)


    out_cloud = ''.join([out_file[:-4], '_masked', out_file[-4:]])
    out_cnt = ''.join([out_file[:-4], '_clear', out_file[-4:]])
    out_date = ''.join([out_file[:-4], '_date', out_file[-4:]])

    create_ds(out_file, extent, gdal.GDT_Int16, resource, driver, bands=6)
    create_ds(out_cloud, extent, gdal.GDT_UInt16, resource, driver, bands=1)
    create_ds(out_cnt, extent, gdal.GDT_UInt16, resource, driver, bands=1)
    create_ds(out_date, extent, gdal.GDT_UInt32, resource, driver, bands=1)

    affine = buildaffine(extent[0], extent[1])
    chip_ls = buildrequestls(truth, extent[0], extent[1])
    total = len(chip_ls)

    with mp.Pool(cpu) as pool:
        with tqdm.tqdm(total=total) as pbar:
            for res in pool.imap_unordered(func, chip_ls):
                if res is None:
                    continue

                row_off, col_off = transform_geo(res['x'], res['y'], affine)

                # Write out Median Composite
                for idx, b_data in enumerate(res['median']):
                    write_arr(out_file, b_data, col_off, row_off, idx + 1)

                # Write out Mask File
                write_arr(out_cloud, res['masked'], col_off, row_off, 1)

                # Write out Clear File
                write_arr(out_cnt, res['not-masked'], col_off, row_off, 1)

                # Write out date
                write_arr(out_date, res['dates'], col_off, row_off, 1)

                pbar.update()


def main_median_nomask(shapepath: str, ranges: Iterable, out_file: str, driver: str, cpu: int, region: str, sensors: list, composite: list):
    """
    """
    resource = regionresource(region)
    truth, extent = rasterize(shapepath, resource)

    func = partial(worker,
                   ranges=ranges,
                   resource=resource,
                   func=median_value_nomask,
                   sensors=sensors,
                   composite=composite)

    out_image = ''.join([out_file[:-4], '_median', out_file[-4:]])
    create_ds(out_image, extent, gdal.GDT_Int16, resource, driver, bands=6)
   

    affine = buildaffine(extent[0], extent[1])
    chip_ls = buildrequestls(truth, extent[0], extent[1])
    total = len(chip_ls)

    with mp.Pool(cpu) as pool:
        with tqdm.tqdm(total=total) as pbar:
            for cx, cy, data in pool.imap_unordered(func, chip_ls):
                row_off, col_off = transform_geo(cx, cy, affine)
                #Write out Median Composite
                for idx, b_data in enumerate(data[:6]):
                    # d = mask_chip(truth, b_data, row_off, col_off)
                    write_arr(out_image, b_data, col_off, row_off, idx + 1)
            
                pbar.update()



def main(args: argparse.Namespace):
    ranges = args.ranges.split(',')
    sensors = args.sensors.split(',')

    if args.composite == 'leafon':
        print('Generating Composite with Median Value for Leafon')
        main_median(args.shapefile, ranges, args.outfile, args.driver, args.cpu, args.region, sensors, args.composite)
    elif args.composite == 'leafoff':
        print('Generating Composite with Median Value for Leafoff')
        main_median(args.shapefile, ranges, args.outfile, args.driver, args.cpu, args.region, sensors, args.composite)
    elif args.composite == 'reference':
        print('Generating Composite with Median Value for Reference with PixelQA')
        main_median(args.shapefile, ranges, args.outfile, args.driver, args.cpu, args.region, sensors, args.composite)
    elif args.composite == 'leafoff_nosnow':
        print('Generating Composite with ARD value wiht no snow')
        main_median(args.shapefile, ranges, args.outfile, args.driver, args.cpu, args.region, sensors, args.composite)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Landsat composites using LCMAP services', formatter_class=RawTextHelpFormatter)
    parser.add_argument('shapefile',
                        type=str,
                        help='Path to a shape file to represent the AoI (must use the AEA/WGS84 projection).\n'
                             'Example: my_aoi.shp')
    parser.add_argument('outfile',
                        type=str,
                        help='Output path, including file extension.\n'
                             'Example: my_composite.tif')
    parser.add_argument('ranges',
                        type=str,
                        help='Comma separated list of date ranges to consider.\n'
                             'Example: 2014-05-01/2014-09-01,2015-05-01/2015-09-01,2016-05-01/2016-09-01')
    parser.add_argument('-region',
                        type=str,
                        choices=['cu', 'ak', 'hi'],
                        default='cu',
                        required=False,
                        help='ARD region which the processing takes place.\n'
                             'Default: cu')
    parser.add_argument('-sensors',
                        type=str,
                        default='l8',
                        required=False,
                        help='Comma separated list of sensors to consider, or "all" for all Landsat sensors.\n'
                             'Example: l7,l8')
    parser.add_argument('-driver',
                        type=str,
                        default='HFA',
                        required=False,
                        help='GDAL raster driver to use.\n'
                             'Example: GTiff for .tif, or HFA for .img)\n'
                             'Full list: https://gdal.org/drivers/raster/index.html\nDefault: GTiff')
    parser.add_argument('-cpu',
                        type=int,
                        default=1,
                        required=False,
                        help='Number of sub-processes to spawn to help retrieve and process data.\n'
                             'Default: 1')
    parser.add_argument('-composite',
                        type=str,
                        default='leafon',
                        required=False,
                        help='Type leafon or leafoff or reference')

    args = parser.parse_args()
    main(args)
