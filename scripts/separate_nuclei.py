import os

from pathlib import Path

import numpy as np

import click

from imageio import imread, imsave

from natsort import natsorted

from skimage.filters import threshold_otsu
from skimage.measure import label

from scipy.ndimage.filters import gaussian_filter


# def force_to_scaled_uint8(im2d):

#     im2d = im2d.astype(np.uint8)

#     scaled = (im2d - im2d.min()) / (im2d.max() - im2d.min())

#     return scaled


def im3d_to_fpath(fpath, im3d):

    path = Path(fpath)

    path.mkdir(exist_ok=True, parents=True)

    for i in range(im3d.shape[2]):
        imsave(
            path/'im{:03}.png'.format(i),
            im3d[:,:,i],
            compress_level=0
        )


def im3d_from_fpath(fpath):

    fpath = Path(fpath)

    sorted_file_list = natsorted(os.listdir(fpath))

    ims = [imread(fpath / f) for f in sorted_file_list]

    return np.dstack(ims)


def spike_nuclei(image_series_dirpath):

    im3d = im3d_from_fpath(image_series_dirpath)

    blurred = gaussian_filter(im3d, sigma=5)
    im3d_to_fpath('scratch/blurred', blurred)

    global_thresh = threshold_otsu(blurred)
    thresholded = 255 * (blurred > global_thresh).astype(np.uint8)
    im3d_to_fpath('scratch/thresholded', thresholded)

    connected_components = label(thresholded)

    labels = list(np.unique(connected_components))
    labels.remove(0)

    l = 2

    coords = np.where(connected_components == l)
    rcoords, ccoords, zcoords = coords
    rmin = rcoords.min()
    rmax = rcoords.max()
    cmin = ccoords.min()
    cmax = ccoords.max()
    zmin = zcoords.min()
    zmax = zcoords.max()

    p = 5

    mask = np.where(connected_components != l)
    imscaled = 255 * (im3d - im3d.min()) / (im3d.max() - im3d.min())
    # im3d[mask] = 0
    nuc = imscaled[rmin-p:rmax+p,cmin-p:cmax+p,zmin-p:zmax+p]

    im3d_to_fpath('scratch/nuc{}'.format(l), nuc)


@click.command()
@click.argument('image_series_dirpath')
def main(image_series_dirpath):

    spike_nuclei(image_series_dirpath)


if __name__ == '__main__':
    main()
