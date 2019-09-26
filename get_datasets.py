#!/usr/bin/env python

from pathlib import Path
from tensorflow.keras.utils import get_file
from scipy.io import loadmat


datasets_dir = Path('Datasets')
datasets_dir.mkdir(exist_ok=True)

# Flowers
flowers_dir = datasets_dir / 'Flowers'
flowers_dir.mkdir(exist_ok=True)

flowers_images_download = get_file(
    fname='images.tgz',
    origin='http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz',
    cache_subdir=flowers_dir.absolute(),
    extract=True
)

flowers_segments_download = get_file(
    fname='segments.tgz',
    origin='http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102segmentations.tgz',
    cache_subdir=flowers_dir.absolute(),
    extract=True
)

flowers_split_download = get_file(
    fname='setid.mat',
    origin='http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat',
    cache_subdir=flowers_dir.absolute()
)

flower_splits = loadmat(flowers_split_download)
# dump to file
#np.savetxt(flowers_dir / 'train_val_test_split.csv', fmt='%d')

# Faces
faces_dir = datasets_dir / 'Faces'
faces_dir.mkdir(exist_ok=True)

faces_funnel_download = get_file(
    fname='funnels.tgz',
    origin='http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz',
    cache_subdir=faces_dir.absolute(),
    extract=True
)

faces_truth_download = get_file(
    fname='images.tgz',
    origin=('http://vis-www.cs.umass.edu/lfw/part_labels/parts_lfw_funneled_gt_'
            'images.tgz'),
    cache_subdir=faces_dir.absolute(),
    extract=True
)

faces_train_download = get_file(
    fname='train.txt',
    origin='http://vis-www.cs.umass.edu/lfw/part_labels/parts_train.txt',
    cache_subdir=faces_dir.absolute()
)

faces_test_download = get_file(
    fname='test.txt',
    origin='http://vis-www.cs.umass.edu/lfw/part_labels/parts_test.txt',
    cache_subdir=faces_dir.absolute()
)

faces_valid_download = get_file(
    fname='valid.txt',
    origin='http://vis-www.cs.umass.edu/lfw/part_labels/parts_validation.txt',
    cache_subdir=faces_dir.absolute()
)

# Birds
birds_dir = datasets_dir / 'Birds'
birds_dir.mkdir(exist_ok=True)

birds_images_download = get_file(
    fname='images.tgz',
    origin=('http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_'
            '2011.tgz'),
    cache_subdir=birds_dir.absolute(),
    extract=True
)

birds_segments_download = get_file(
    fname='segments.tgz',
    origin=('http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/'
            'segmentations.tgz'),
    cache_subdir=birds_dir.absolute(),
    extract=True
)

birds_split_download = get_file(
    fname='segments.tgz',
    origin=('https://raw.githubusercontent.com/mickaelChen/ReDO/master/'
            'datasplits/cub/train_val_test_split.txt'),
    cache_subdir=birds_dir.absolute()
)
