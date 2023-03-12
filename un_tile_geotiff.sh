#!/bin/bash

gdalbuildvrt -allow_projection_difference index.vrt *_CLASSIFICATION_MAP.tif 
gdal_translate index.vrt weed_map.tif -co COMPRESS=LZW -co BIGTIFF=YES -a_nodata 0


