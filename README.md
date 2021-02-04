# composite
Image Composites as an input to Land Cover Classification

# hypothesis

- The cloud will offer efficiencies and opportunities to accelerate science
- lcmap will be ported and likely re-engineered in the cloud
- Adhoc science R&D should start getting ready for these paradigm shifting events

# hybrid approach posssible

- compositing using ubiquitous cloud assets
- send composites to Denali for heavy cpu - Land Cover - Characterization 
    - uses compute intense processes including promising AI techniques


# approach

- Play with Landsat-pds data
- speculate on tiling and scaling approaches for composites, NLCD and 
- monitor cloud based assets in COGs as they onfold - track Albers scenes and Tiles
- Play with Sentinel-2 COGS in AWS Public Data Registry
- Be aware of HLS and the decisions made on projections and gridding
    - UTM
    - MGRS
    

https://en.wikipedia.org/wiki/Military_Grid_Reference_System#/media/File:Universal_Transverse_Mercator_zones.svg

# steps
0. - Everyone has access to a science sandbox - mini-pangeo - at 10.12.68.150 (neal's account - via vpn)
    - also the pangeo.chs.usgs.gov - (can also get a pangeo bucket)
    - onramp to cloud ec2
    
1. get familiar with cloud assets - searchable via STAC
    - landsat-pds
    - `sentinel`
    - `landsat-sr-albers`
    - potentially HLS - which is curated by NASA LPDAAC and uses CMR-CMR-STAC API 
        - POC - Cole Krehbiel - Aaron Friesz
https://lpdaac.usgs.gov/resources/e-learning/
https://lpdaac.usgs.gov/resources/e-learning/getting-started-cloud-native-hls-data-python/

    
2. a quick side by side demo of animations sentinel and landsat
 
3. converting/slicing xarray data in a format for compositing - time slices for
     - reds greens blues nirs swir1s swir2s pixel_qas
     
4. There may be synergies with LCMAPs cloud efforts
    - Kelcy - Jon Morton - Jeff Briden -- (mangage the rates from limits ...)
    
5. Explore Scaling 
    - Docker (swarm)
    - Kubernetes ... usual suspects ... (like mesos)
    - AWS batch
    - Denali - slurm - (preallocate - what you want for cpu ,memory)
        - 9000 actual cores out of 18000(hyperthreading) total - (Shrub Back In Time) Cubist 
        - windows (3by 32 cores) 100 times fater 1/2 hour for 1.5 TBytes of data

6. 


