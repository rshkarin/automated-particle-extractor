import argparse
import os
import sys
import timeit
import warnings
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import skimage.measure as measure
import skimage.util as util
import skimage.exposure as exp
import skimage.filters as filters
import matplotlib.pyplot as plt
from PIL import Image

_MEASUREMENTS = {
    'Label': 'label',
    'Area': 'area',
    'Perimeter': 'perimeter'
}

_MEASUREMENTS_VALS = _MEASUREMENTS.values()

class Timer(object):
    def __init__(self):
        self.start = timeit.default_timer()

    def elapsed(self, title):
        self.stop = timeit.default_timer()
        diff = self.stop - self.start
        print "Elapsed time (%s): %f sec ~= %f min" % (title, diff, diff / 60.)

def write_as_raw(data, sample_name, output_dir, prefix=None):
    bits = -1
    if data.dtype == np.int32 or data.dtype == np.float32:
        bits = 32
    elif data.dtype == np.uint8 or data.dtype == np.bool:
        bits = 8

    size = data.shape[::-1]
    output_filename = '{0}_{1}bit_{2}x{3}x{4}.raw'.format(sample_name, bits, *size) if prefix is None \
                        else '{0}_{1}_{2}bit_{3}x{4}x{5}.raw'.format(sample_name, prefix, bits, *size)
    data.tofile(os.path.join(output_dir, output_filename))

def object_counter(stack_binary_data):
    t = Timer()
    print 'Object counting - Labeling...'
    labeled_stack, num_labels = ndi.measurements.label(stack_binary_data)
    t.elapsed('Object counting - Labeling ({0} labels)...'.format(num_labels))

    objects_stats = pd.DataFrame(columns=_MEASUREMENTS_VALS)

    t = Timer()
    print 'Object counting - Stats gathering...'
    for slice_idx in np.arange(labeled_stack.shape[0]):
        for region in measure.regionprops(labeled_stack[slice_idx]):
            objects_stats = objects_stats.append({_measure: region[_measure] \
                                        for _measure in _MEASUREMENTS_VALS}, \
                                            ignore_index=True)
    t.elapsed('Object counting - Stats gathering...')

    t = Timer()
    print 'Object counting - Stats grouping...'
    objects_stats = objects_stats.groupby('label', as_index=False).sum()
    t.elapsed('Object counting - Stats grouping...')

    return objects_stats, labeled_stack

def open_data(sample_dir, place_type, reco_folder='reconstructed'):
    input_data = None

    if place_type == 'CTLab':
        sample_path = os.path.join(sample_dir, reco_folder)
        files = [f for f in os.listdir(sample_path) if f.endswith(".tif")]
        _image = np.array(Image.open(os.path.join(sample_path, files[0])))
        height, width = _image.shape
        input_data = np.zeros((len(files), height, width), dtype=np.float32)

        for i,f in enumerate(files):
            input_data[i] = np.array(Image.open(os.path.join(sample_path, f)))

    return input_data

def process_batch(samples, input_dir, place_type, verbose=True, output_folder='Analysis'):
    for sample in samples:
        print '###### Processing of {0}'.format(sample)
        t = Timer()
        print 'Data processing - Opening and filtering...'
        input_data = open_data(os.path.join(input_dir, sample), place_type)
        filtered_input_data = ndi.filters.median_filter(input_data, size=(2,2,2))
        filtered_input_data = ndi.filters.gaussian_filter(input_data, sigma=1)
        t.elapsed('Data processing - Opening and filtering...')

        t = Timer()
        print 'Data processing - Binarizing...'
        thresholded_data = np.zeros_like(input_data)
        n_slices = input_data.shape[0]

        for idx,data_slice in enumerate(input_data):
            if idx % 100 == 0 or idx == (n_slices - 1):
                print 'Slice {0}/{1}'.format(idx, input_data.shape[0])

            #global histogram stretching
            p2, p98 = np.percentile(data_slice, (2, 98))
            data_slice = exp.rescale_intensity(data_slice, out_range=(p2, p98))
            threshold_value = filters.threshold_otsu(data_slice)
            thresholded_data[idx] = (data_slice > threshold_value)

        t.elapsed('Data processing - Binarizing...')

        t = Timer()
        print 'Data processing - Noise filtering...'
        thresholded_data = util.img_as_ubyte(thresholded_data)
        thresholded_data = ndi.morphology.binary_opening(thresholded_data, structure=np.ones((2,2,2)), iterations=1)
        thresholded_data = ndi.filters.median_filter(thresholded_data, size=(3,3,3))
        t.elapsed('Data processing - Noise filtering...')

        #object counting
        objects_stats, labeled_data = object_counter(thresholded_data)

        t = Timer()
        print 'Data saving - Stats and data saving...'
        output_dir = os.path.join(input_dir, sample, output_folder)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print labeled_data.dtype
        print thresholded_data.dtype

        write_as_raw(labeled_data, sample, output_dir, prefix='labels')
        write_as_raw(thresholded_data, sample, output_dir, prefix='mask')
        objects_stats.to_csv(os.path.join(output_dir, 'particles_stats_{0}.csv'.format(sample)))
        t.elapsed('Data saving - Stats and data saving...')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", help="path to folders with data", required=True)
    parser.add_argument("-s", "--samples", nargs='+', help="names of sample folders (e.g. scan_01 scan_02 snan_03)", required=True)
    parser.add_argument("-t", "--type", default="CTLab", help="palce where data are obtained (CTLab, Beamline)", required=True)
    parser.add_argument("-v", "--verbose", action="store_true", help="show log")
    args = parser.parse_args()

    print args.input_dir
    print args.samples
    print args.type
    print args.verbose

    process_batch(args.samples, args.input_dir, args.type, verbose=args.verbose)

if __name__ == "__main__":
    sys.exit(main())
