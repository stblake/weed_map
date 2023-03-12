
# Break a (large) geotiff into a grid of smaller geotiffs

# Sam Blake, started 11 March, 2023.


import os
from osgeo import gdal
from argparse import ArgumentParser



def tile_geotiff():

    # Read command line args. 
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", dest="input_filename", default="img.tif")
    parser.add_argument("-n_tiles_x", "--n_tiles_x", dest="n_tiles_x")
    parser.add_argument("-n_tiles_y", "--n_tiles_y", dest="n_tiles_y")
    args = parser.parse_args()

    if not os.path.exists(args.input_filename):
        print(f'input_filename = {args.input_filename} does not exist. Exiting...')
        return 1

    output_filename = args.input_filename.replace('.tif', '')

    ds = gdal.Open(args.input_filename)
    band = ds.GetRasterBand(1)
    nx = band.XSize
    ny = band.YSize
 
    tile_size_x = nx//int(args.n_tiles_x)
    tile_size_y = ny//int(args.n_tiles_y)
    print(f'tile_size_x, tile_size_y = {tile_size_x}, {tile_size_y}')

    k = 0
    for i in range(0, nx, tile_size_x):
        for j in range(0, ny, tile_size_y):
            k += 1
            print(f'Processing tile {k}...')
            com_string = "gdal_translate -of GTIFF -srcwin " + str(i) + ", " + \
                str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + \
                str(args.input_filename) + " " + str(output_filename) + "_tile_" + str(k) + ".tif"
            os.system(com_string)

    return 1



if __name__ == "__main__":
    tile_geotiff()
