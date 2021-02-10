
from setuptools import setup

setup(name='compositeLib',
      maintainer='Tony b',
      maintainer_email='tonybutzer@gmail.com',
      version='1.0.1',
      description='helper functions for invasive data wrangling',
      packages=[
          'compositeLib',
      ],
      install_requires=[
          'xarray',
          'folium',
          'shapely',
          'pyproj',
      ],

      )
