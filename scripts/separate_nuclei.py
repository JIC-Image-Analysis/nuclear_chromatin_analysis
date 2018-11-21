import os

from pathlib import Path

import numpy as np

import click

from imageio import imread, imsave, mimsave

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


def im3d_to_tiff(fpath, im3d):

    root, ext = os.path.splitext(fpath)

    assert ext in ['.tif', '.tiff']

    # We use row, col, z, but mimsave expects z, row, col
    transposed = np.transpose(im3d, axes=[2, 0, 1])

    mimsave(fpath, transposed)


def im3d_from_dirpath(fpath):

    fpath = Path(fpath)

    sorted_file_list = natsorted(os.listdir(fpath))

    ims = [imread(fpath / f) for f in sorted_file_list]

    return np.dstack(ims)


def find_name_root(dirpath):

    return os.path.basename(os.path.normpath(dirpath))


def find_objects(im3d):

    blurred = gaussian_filter(im3d, sigma=5)
    global_thresh = threshold_otsu(blurred)
    thresholded = 255 * (blurred > global_thresh).astype(np.uint8)
    connected_components = label(thresholded)

    return connected_components


def scale_to_uint8(im3d):

    imscaled = im3d.astype(np.float32)
    imscaled = 255 * (imscaled - imscaled.min()) / (imscaled.max() - imscaled.min())
    imscaled = imscaled.astype(np.uint8)

    return imscaled


def find_bounding_cube(coords):

    rcoords, ccoords, zcoords = coords

    rmin = rcoords.min()
    rmax = rcoords.max()
    cmin = ccoords.min()
    cmax = ccoords.max()
    zmin = zcoords.min()
    zmax = zcoords.max()

    return (rmin, rmax, cmin, cmax, zmin, zmax)


def apply_bc_to_image(im3d, bc, xypad, zpad):

    rdim, cdim, zdim = im3d.shape

    rmin, rmax, cmin, cmax, zmin, zmax = bc
    p_zmin = max(zmin - zpad, 0)
    p_zmax = min(zmax + zpad, zdim)

    return im3d[rmin-xypad:rmax+xypad,cmin-xypad:cmax+xypad,p_zmin:p_zmax]


def extract_individual_nuclei(image_series_dirpath, output_dirpath):

    name_root = find_name_root(image_series_dirpath)

    im3d = im3d_from_dirpath(image_series_dirpath)

    connected_components = find_objects(im3d)
    labels = list(np.unique(connected_components))
    labels.remove(0)

    imscaled = scale_to_uint8(im3d)
    for l in labels:
        coords = np.where(connected_components == l)
        print(len(coords[0]) * 0.04 * 0.04 * 0.11)
        bc = find_bounding_cube(coords)
        nuclear_region = apply_bc_to_image(imscaled, bc, xypad=20, zpad=8)
        fname = "{}-nucleus{}.tif".format(name_root, l)
        fpath = os.path.join(output_dirpath, fname)
        im3d_to_tiff(fpath, nuclear_region)


def spike_nuclei(image_series_dirpath):

    im3d = im3d_from_dirpath(image_series_dirpath)

    blurred = gaussian_filter(im3d, sigma=5)
    im3d_to_tiff('scratch/blurred.tif', blurred)

    global_thresh = threshold_otsu(blurred)
    thresholded = 255 * (blurred > global_thresh).astype(np.uint8)
    im3d_to_tiff('scratch/thresholded.tif', thresholded)

    connected_components = label(thresholded)

    labels = list(np.unique(connected_components))
    labels.remove(0)

    rdim, cdim, zdim = im3d.shape
    xyp = 20
    zp = 8
    imscaled = im3d.astype(np.float32)
    imscaled = 255 * (imscaled - imscaled.min()) / (imscaled.max() - imscaled.min())
    imscaled = imscaled.astype(np.uint8)

    for l in labels:
        coords = np.where(connected_components == l)
        rcoords, ccoords, zcoords = coords
        rmin = rcoords.min()
        rmax = rcoords.max()
        cmin = ccoords.min()
        cmax = ccoords.max()
        zmin = zcoords.min()
        zmax = zcoords.max()

        mask = np.where(connected_components != l)

        # FIXME - masking really needs the ROI dilated first
        # imscaled[mask] = 0
        p_zmin = max(zmin - zp, 0)
        p_zmax = min(zmax + zp, zdim)
        print(p_zmin, p_zmax)
        nuc = imscaled[rmin-xyp:rmax+xyp,cmin-xyp:cmax+xyp,p_zmin:p_zmax]

        im3d_to_tiff('scratch/nucleus{}.tif'.format(l), nuc)


@click.command()
@click.argument('image_series_dirpath')
@click.argument('output_dirpath')
def main(image_series_dirpath, output_dirpath):

    extract_individual_nuclei(image_series_dirpath, output_dirpath)


if __name__ == '__main__':
    main()
