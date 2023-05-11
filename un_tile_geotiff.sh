#!/bin/bash

gdalbuildvrt -allow_projection_difference index.vrt *_WEED_MAP.tif 
gdal_translate index.vrt weed_map.tif -co COMPRESS=LZW -co BIGTIFF=YES -a_nodata 0


gdalbuildvrt -allow_projection_difference index.vrt *_SPRAY_MAP.tif 
gdal_translate index.vrt spray_map.tif -co COMPRESS=LZW -co BIGTIFF=YES -a_nodata 0
