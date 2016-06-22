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
import skimage.restoration as rest
import skimage.filters as filters
import skimage.feature as feature
import skimage.color as color
import skimage.morphology as morphology
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

def write_as_2d_raw(data, sample_name, output_dir, prefix=None):
    bits = -1
    if data.dtype == np.int32 or data.dtype == np.float32:
        bits = 32
    elif data.dtype == np.uint8 or data.dtype == np.bool:
        bits = 8

    size = data.shape[::-1]
    output_filename = '{0}_{1}bit_{2}x{3}.raw'.format(sample_name, bits, *size) if prefix is None \
                        else '{0}_{1}_{2}bit_{3}x{4}.raw'.format(sample_name, prefix, bits, *size)
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

def object_counter_by_labels(labeled_stack):
    t = Timer()
    print 'Object counting - Labeling...'
    t.elapsed('Object counting - Labeling...')

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

def renyi_entropy_thresholding(im, alpha=3):
  hist = exp.histogram(im)
  hist_float = [float(i) for i in hist[0]]
  pdf = hist_float / np.sum(hist_float)
  cumsum_pdf = np.cumsum(pdf)

  s, e = 0, 255
  scalar = 1. / (1. - alpha)
  eps = np.spacing(1)

  rr = e - s
  h1 = np.zeros((rr, 1))
  h2 = np.zeros((rr, 1))

  for ii in range(1, rr):
      iidash = ii + s

      temp1 = np.power(pdf[1:iidash] / cumsum_pdf[iidash], scalar)
      h1[ii] = np.log(np.sum(temp1) + eps)

      temp2 = np.power(pdf[iidash+1:255] / (1. - cumsum_pdf[iidash]), scalar)
      h2[ii] = np.log(np.sum(temp2) + eps)

  T = h1 + h2
  T = -T * scalar
  location = T.argmax(axis=0)

  return location

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

def process_batch(samples, input_dir, place_type, verbose=True, output_folder='Analysis_nlm_temp'):
    for sample in samples:
        print '###### Processing of {0}'.format(sample)
        t = Timer()
        print 'Data processing - Opening and filtering...'
        input_data = open_data(os.path.join(input_dir, sample), place_type)
        output_dir = os.path.join(input_dir, sample, output_folder)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filtered_input_data = ndi.filters.median_filter(input_data, size=(2,2,2))
        filtered_input_data = ndi.filters.gaussian_filter(input_data, sigma=1)
        #filtered_input_data = rest.denoise_nl_means(input_data, patch_size=5, patch_distance=7, h=0.16, multichannel=False, fast_mode=False)
        t.elapsed('Data processing - Opening and filtering...')

        t = Timer()
        print 'Data processing - Binarizing...'
        thresholded_data = np.zeros_like(filtered_input_data)
        n_slices = input_data.shape[0]

        for idx,data_slice in enumerate(filtered_input_data):
            if idx % 100 == 0 or idx == (n_slices - 1):
                print 'Slice {0}/{1}'.format(idx, n_slices)

            #global histogram stretching
            p1, p2 = np.percentile(data_slice, 0.2), np.percentile(data_slice, 99.8)
            data_slice = exp.rescale_intensity(data_slice, in_range=(p1, p2))
            threshold_value = filters.threshold_otsu(data_slice)
            thresholded_data[idx] = (data_slice > threshold_value)

            thresholded_data[idx] = data_slice

        t.elapsed('Data processing - Binarizing...')

        #write_as_raw(thresholded_data, sample, output_dir, prefix='filtered_data_nothresh')
        #continue

        t = Timer()
        print 'Data processing - Noise filtering...'
        thresholded_data = util.img_as_ubyte(thresholded_data)
        #thresholded_data = ndi.morphology.binary_opening(thresholded_data, structure=np.ones((2,2,2)), iterations=1)
        #thresholded_data = ndi.filters.median_filter(thresholded_data, size=(5,5,5))
        t.elapsed('Data processing - Noise filtering...')

        #object counting
        objects_stats, labeled_data = object_counter(thresholded_data)

        t = Timer()
        print 'Data saving - Stats and data saving...'

        write_as_raw(input_data, sample, output_dir, prefix='original')
        write_as_raw(filtered_input_data, sample, output_dir, prefix='filtered_data')
        write_as_raw(labeled_data, sample, output_dir, prefix='labels')
        write_as_raw(thresholded_data, sample, output_dir, prefix='mask')
        objects_stats.to_csv(os.path.join(output_dir, 'particles_stats_{0}.csv'.format(sample)))

        t.elapsed('Data saving - Stats and data saving...')

def main2():
    data_path = "Z:\\tomo\\rshkarin\\SvetaRawData\\{0}\\Analysis_temp"
    names = ['scan_0008_filtered_data_thresh_8bit_1536x1536x786', 'scan_0010_filtered_data_thresh_8bit_1536x1536x786', 'scan_0005_filtered_data_thresh_8bit_1536x1536x826', 'scan_0007_filtered_data_thresh_8bit_1536x1536x787']
    sizes = [(768,1536,1536,), (768,1536,1536,), (826,1536,1536,), (787,1536,1536)]
    samples = ['scan_0008', 'scan_0010', 'scan_0005', 'scan_0007']

    for name, size, sample in zip(names, sizes, samples):
        dpath = data_path.format(sample)
        print dpath
        input_data_path = os.path.join(dpath, name) + '.raw'
        input_data = np.memmap(input_data_path, shape=size, dtype=np.uint8)
        input_data = input_data / 255
        objects_stats, labeled_data = object_counter(input_data)
        objects_stats.to_csv(os.path.join(dpath, 'particles_stats_{0}.csv'.format(name)))

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

def separate_particles():
    data_path = "Z:\\tomo\\rshkarin\\SvetaRawData\\{0}\\Analysis_temp"
    #names = ['scan_0010_filtered_data_thresh_8bit_1536x1536x786']
    names = ['scan_0010_filtered_data_thresh_8bit_1536x1536x1']
    sizes = [(1536,1536,)]
    samples = ['scan_0010']

    for name, size, sample in zip(names, sizes, samples):
        dpath = data_path.format(sample)
        print dpath
        input_data_path = os.path.join(dpath, name) + '.raw'
        input_data = np.memmap(input_data_path, shape=size, dtype=np.uint8)

        #input_data = ndi.morphology.binary_opening(input_data, structure=np.ones((5,5,1)), iterations=1)
        #write_as_raw(input_data, sample, dpath, prefix='opened_filtered_data_thresh')

        print 'ndimage.interpolation.zoom'
        zoomed_data = ndi.interpolation.zoom(input_data, 0.5, order=0)
        markers_zoomed_data = morphology.remove_small_objects(ndi.label(zoomed_data)[0], 15)
        markers_zoomed_data = ndi.filters.median_filter(markers_zoomed_data, size=(4,4))
        markers_zoomed_data = markers_zoomed_data > 0
        markers_zoomed_data = ndi.morphology.binary_fill_holes(markers_zoomed_data, structure=np.ones((3,3)))
        markers_zoomed_data_labels = ndi.label(markers_zoomed_data)[0]
        # markers_zoomed_data = markers_zoomed_data > 0
        # markers_zoomed_data = markers_zoomed_data.astype(np.uint8)
        # markers_zoomed_data = ndi.morphology.binary_fill_holes(markers_zoomed_data, structure=np.ones((1,3,3)))
        # #zoomed_data = ndi.filters.median_filter(zoomed_data, size=(1,4,4))
        # markers_zoomed_data = ndi.label(zoomed_data)[0]
        # markers_zoomed_data = morphology.remove_small_objects(markers_zoomed_data, 15)
        # markers_zoomed_data = ndi.filters.median_filter(markers_zoomed_data, size=(1,4,4))
        write_as_2d_raw(markers_zoomed_data, sample, dpath, prefix='binary_fill_holes_filtered_data_thresh')
        write_as_2d_raw(markers_zoomed_data_labels, sample, dpath, prefix='labels_binary_fill_holes_filtered_data_thresh')

        print 'distance_transform_edt'
        distance = ndi.distance_transform_edt(markers_zoomed_data)
        write_as_2d_raw(distance, sample, dpath, prefix='distance_transform_edt_data_thresh')

        print 'peak_local_max'
        #local_maxi = feature.peak_local_max(distance, indices=False, footprint=np.ones((3, 3, 3)), labels=zoomed_data)
        #local_maxi = feature.peak_local_max(distance, indices=False, footprint=np.ones((1, 3, 3)), labels=markers_zoomed_data)
        local_maxi = feature.peak_local_max(distance, indices=False,  footprint=morphology.disk(3), labels=markers_zoomed_data).astype(np.uint8)
        write_as_2d_raw(local_maxi, sample, dpath, prefix='peak_local_max_filtered_data_thresh')

        print 'label(local_maxi)[0]'
        markers = ndi.label(local_maxi)[0]

        print 'morphology.watershed'
        labels = morphology.watershed(-distance, markers, mask=markers_zoomed_data)

        print 'write_as_raw'
        write_as_2d_raw(labels, sample, dpath, prefix='separated_particles')

def separate_particles2():
    data_path = "Z:\\tomo\\rshkarin\\SvetaRawData\\{0}\\Analysis_temp"
    #names = ['scan_0010_filtered_data_thresh_8bit_1536x1536x786']
    #names = ['scan_0010_slice509_filtered_data_nothresh_32bit_1536x1536']
    names = ['scan_0010_filtered_data_nothresh_32bit_768x768x393']
    #sizes = [(1536,1536,)]
    #sizes = [(786,1536,1536,)]
    sizes = [(393,768,768,)]
    samples = ['scan_0010']

    for name, size, sample in zip(names, sizes, samples):
        dpath = data_path.format(sample)
        print dpath
        input_data_path = os.path.join(dpath, name) + '.raw'
        input_data = np.memmap(input_data_path, shape=size, dtype=np.float32)

        p1, p2 = np.percentile(input_data, 0.2), np.percentile(input_data, 99.8)

        print 'exp.rescale_intensity'
        data_slice = exp.rescale_intensity(input_data, in_range=(p1, p2))

        print 'util.img_as_ubyte'
        data_slice = util.img_as_ubyte(data_slice)
        #write_as_2d_raw(data_slice, sample, dpath, prefix='intensity_rescaled_slice509')
        write_as_raw(data_slice, sample, dpath, prefix='intensity_rescaled')

        print 'filters.rank.median'
        #denoised = filters.rank.median(data_slice, morphology.disk(5))
        #denoised = ndi.filters.median_filter(data_slice, size=(3,3,3))
        denoised = filters.gaussian(denoised, 2, mode='nearest')

        #write_as_2d_raw(denoised, sample, dpath, prefix='denoised_intensity_rescaled_slice509')
        write_as_raw(denoised, sample, dpath, prefix='denoised_intensity_rescaled')

        print 'filters.rank.gradient'
        #markers = filters.rank.gradient(denoised, morphology.disk(5)) < 25

        markers = np.gradient(denoised) < 25

        write_as_raw(markers, sample, dpath, prefix='gradient_intensity_rescaled')

        markers = ndi.label(markers)[0]

        print 'filters.rank.gradient - 2'
        #gradient = filters.rank.gradient(denoised, morphology.disk(1))
        gradient = np.gradient(denoised)
        #write_as_2d_raw(gradient, sample, dpath, prefix='gradient2_slice509')
        write_as_raw(gradient, sample, dpath, prefix='gradient2')

        print 'filters.threshold_otsu'
        masked_data = np.zeros_like(denoised)
        for i,d_slice in enumerate(denoised):
            threshold_value = filters.threshold_otsu(d_slice)
            masked_data[i] = (d_slice > threshold_value)

        print 'morphology.watershed'
        labels = morphology.watershed(gradient, markers, mask=masked_data)

        # write_as_2d_raw(markers, sample, dpath, prefix='local_grad_slice509')
        # write_as_2d_raw(labels, sample, dpath, prefix='watershed_slice509')
        # write_as_2d_raw(masked_data, sample, dpath, prefix='labels_slice509')
        #
        write_as_raw(markers, sample, dpath, prefix='local_grad')
        write_as_raw(labels, sample, dpath, prefix='watershed')
        write_as_raw(masked_data, sample, dpath, prefix='labels')

def separate_particles2():
    data_path = "Z:\\tomo\\rshkarin\\SvetaRawData\\{0}\\Analysis_temp"
    names = ['scan_0010_filtered_data_nothresh_32bit_768x768x393']
    # names = ['scan_0010_filtered_data_nothresh_32bit_384x384x196']
    # sizes = [(196,384,384,)]
    sizes = [(393,768,768,)]
    samples = ['scan_0010']

    for name, size, sample in zip(names, sizes, samples):
        dpath = data_path.format(sample)
        print dpath
        input_data_path = os.path.join(dpath, name) + '.raw'
        input_data = np.memmap(input_data_path, shape=size, dtype=np.float32)

        p1, p2 = np.percentile(input_data, 0.2), np.percentile(input_data, 99.8)

        print 'exp.rescale_intensity'
        data_slice = exp.rescale_intensity(input_data, in_range=(p1, p2))

        print 'util.img_as_ubyte'
        data_slice = util.img_as_ubyte(data_slice)
        write_as_raw(data_slice, sample, dpath, prefix='intensity_rescaled')

        print 'filters.gaussian'
        denoised = ndi.filters.median_filter(data_slice, size=3)
        write_as_raw(denoised, sample, dpath, prefix='denoised_intensity_rescaled')

        print 'filters.threshold_otsu'
        masked_data = np.zeros_like(denoised)
        for i,d_slice in enumerate(denoised):
            threshold_value = filters.threshold_otsu(d_slice)
            masked_data[i] = (d_slice > threshold_value)

        # print 'filters.rank.gradient'
        # gradient_markers = np.zeros_like(denoised)
        # for i,d_slice in enumerate(denoised):
        #     gradient_markers[i] = filters.rank.gradient(d_slice, morphology.disk(2)) < 25
        #
        # write_as_raw(gradient_markers, sample, dpath, prefix='gradient_markers')
        # gradient_markers = ndi.label(gradient_markers)[0]
        #
        # print 'filters.rank.gradient - 2'
        # gradient = np.zeros_like(denoised)
        # for i,d_slice in enumerate(denoised):
        #     gradient[i] = filters.rank.gradient(d_slice, morphology.disk(2))
        #
        # write_as_raw(gradient, sample, dpath, prefix='gradient')

        #################################################################################

        # print 'distance_transform_edt'
        # distance = ndi.distance_transform_edt(masked_data)
        # write_as_raw(distance, sample, dpath, prefix='distance_transform_edt')
        #
        # print 'peak_local_max'
        # local_maxi = feature.peak_local_max(distance, indices=False,  footprint=np.ones((30,30,30))).astype(np.uint8)
        # write_as_raw(local_maxi, sample, dpath, prefix='peak_local_max_filtered_data_thresh')
        #
        # print 'label(local_maxi)[0]'
        # markers = ndi.label(local_maxi)[0]
        #
        # print 'morphology.watershed'
        # labels = morphology.watershed(-distance, markers, mask=masked_data)

        #################################################################################

        markers = ndi.morphology.binary_erosion(masked_data, structure=np.ones((3,3,3)), iterations=6).astype(np.uint8)
        write_as_raw(markers, sample, dpath, prefix='MARKERS')

        print 'label(local_maxi)[0]'
        markers = ndi.label(markers)[0]

        print 'filters.rank.gradient - 2'
        gradient = np.zeros_like(denoised)
        for i,d_slice in enumerate(denoised):
            gradient[i] = filters.rank.gradient(d_slice, morphology.disk(3))

        write_as_raw(gradient, sample, dpath, prefix='GRADIENT')

        print 'morphology.watershed'
        labels = morphology.watershed(gradient, markers, mask=masked_data)

        # print 'morphology.watershed'
        # labels = morphology.watershed(gradient, gradient_markers, mask=masked_data)

        print 'write_as_raw'
        write_as_raw(labels, sample, dpath, prefix='separated_particles')


        objects_stats, labeled_data = object_counter_by_labels(labels)

        t = Timer()
        print 'Data saving - Stats and data saving...'
        objects_stats.to_csv(os.path.join(dpath, 'particles_stats_watershed_{0}.csv'.format(sample)))

def separate_particles3():
    data_path = "Z:\\tomo\\rshkarin\\SvetaRawData\\{0}\\Analysis_temp"
    names = ['scan_0012_filtered_data_nothresh_32bit_768x768x393']
    # names = ['scan_0012_filtered_data_nothresh_32bit_1536x1536x40']
    #names = ['scan_0012_filtered_data_nothresh_32bit_1536x1536x786']
    sizes = [(393,768,768)]
    #sizes = [(30,768,768)]
    #sizes = [(768,1536,1536)]
    #sizes = [(40,1536,1536)]
    samples = ['scan_0012']

    for name, size, sample in zip(names, sizes, samples):
        dpath = data_path.format(sample)
        print dpath
        input_data_path = os.path.join(dpath, name) + '.raw'
        input_data = np.memmap(input_data_path, shape=size, dtype=np.float32)

        p1, p2 = np.percentile(input_data, 0.2), np.percentile(input_data, 99.8)

        print 'exp.rescale_intensity'
        data_slice = exp.rescale_intensity(input_data, in_range=(p1, p2), out_range=(.0, 1.))
        write_as_raw(data_slice, sample, dpath, prefix='intensity_rescaled')

        print 'util.img_as_ubyte'
        data_slice = util.img_as_ubyte(data_slice)
        write_as_raw(data_slice, sample, dpath, prefix='img_as_ubyte')

        print 'filters.median_filter'
        denoised = ndi.filters.median_filter(data_slice, size=2)
        write_as_raw(denoised, sample, dpath, prefix='denoised_intensity_rescaled')

        print 'filters.threshold_otsu'
        masked_data = np.zeros_like(denoised)
        for i,d_slice in enumerate(denoised):
            threshold_value = filters.threshold_otsu(d_slice)
            masked_data[i] = (d_slice > threshold_value)

        print 'morphology.binary_closing'
        masked_data = ndi.morphology.binary_opening(masked_data, structure=np.ones((2,2,2)), iterations=1).astype(np.uint8)
        #masked_data = ndi.morphology.binary_fill_holes(masked_data, structure=np.ones((5,5,5))).astype(np.uint8)
        masked_data = morphology.remove_small_holes(masked_data, min_size=500)
        write_as_raw(masked_data, sample, dpath, prefix='remove_small_holes')

        print 'distance_transform_edt'
        distance = ndi.distance_transform_edt(masked_data)
        write_as_raw(distance, sample, dpath, prefix='distance_transform_edt')

        print 'peak_local_max'
        local_maxi = feature.peak_local_max(distance, indices=False,  footprint=np.ones((8,8,8))).astype(np.uint8)
        write_as_raw(local_maxi, sample, dpath, prefix='peak_local_max_filtered_data_thresh')

        print 'label(local_maxi)[0]'
        markers = ndi.label(local_maxi)[0]

        print 'morphology.watershed'
        labels = morphology.watershed(-distance, markers, mask=masked_data)
        ########################################################################################
        #
        # print 'label(local_maxi)[0]'
        # markers = ndi.label(masked_data)[0]
        #
        # print 'filters.rank.gradient - 2'
        # gradient = np.zeros_like(denoised)
        # for i,d_slice in enumerate(denoised):
        #     gradient[i] = filters.rank.gradient(d_slice, morphology.disk(1))
        #
        # write_as_raw(gradient, sample, dpath, prefix='GRADIENT')
        #
        # print 'morphology.watershed'
        # labels = morphology.watershed(gradient, markers, mask=masked_data)
        #
        image_label_overlay = color.label2rgb(labels[15], image=exp.rescale_intensity(input_data[15], in_range='image', out_range=(.0, 1.)))
        plt.imshow(image_label_overlay, interpolation='nearest')
        plt.show()


        print 'write_as_raw'
        write_as_raw(labels, sample, dpath, prefix='separated_particles')

if __name__ == "__main__":
    #sys.exit(main())
    separate_particles3()
