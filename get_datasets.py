#!/usr/bin/env python

"""Script to get the three datasets we are using. The general setup is to have
four files/folders per each dataset (i) the images (ii) the segmentations
(iii) the paths file and (iv) the split file. To keep it consistent all the
datasets keep the same structure and naming scheme which is
Datasets/
-- DatasetName/
-- -- images/
-- -- -- image_0001.jpg
-- -- -- ...
-- -- labels/
-- -- -- label_001.jpg
-- -- -- ...
-- -- paths.txt
-- -- train_val_test_split.txt

the split file contains two fields per line, the number of the image, and the
type (train, test, validate) which is defined in datasets.Dataset.SPLIT_KEYS
In paths.txt there are two fields per line, the number of the image and the
path assuming the common structure shown above.
"""

from pathlib import Path
from tensorflow.keras.utils import get_file
from scipy.io import loadmat
from operator import itemgetter
from argparse import ArgumentParser
from sys import argv

import datasets


def parse_download_args():
    parser = ArgumentParser()
    parser.add_argument('-f', '--flowers', action='store_true')
    parser.add_argument('-b', '--birds', action='store_true')
    parser.add_argument('-t', '--faces', action='store_true')
    return parser.parse_args(argv[1:])


def configure_flowers(datasets_dir):
    flowers_dir = datasets_dir / 'Flowers'
    flowers_dir.mkdir(exist_ok=True)

    get_file(
        fname='102flowers.tgz',
        origin=('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/'
                '102flowers.tgz'),
        cache_subdir=flowers_dir.absolute(),
        extract=True
    )

    get_file(
        fname='102segmentations.tgz',
        origin=('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/'
                '102segmentations.tgz'),
        cache_subdir=flowers_dir.absolute(),
        extract=True
    )

    flowers_split_download = get_file(
        fname='setid.mat',
        origin='http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat',
        cache_subdir=flowers_dir.absolute()
    )

    flowers_splits = loadmat(flowers_split_download)

    # dump to file
    add_to_inds = lambda inds, type, ids: inds.extend(
        [(id, type) for id in ids])

    flowers_split_inds = []
    add_to_inds(flowers_split_inds, datasets.Dataset.SPLIT_KEYS['training'],
                flowers_splits['trnid'][0].tolist())
    add_to_inds(flowers_split_inds, datasets.Dataset.SPLIT_KEYS['test'],
                flowers_splits['tstid'][0].tolist())
    add_to_inds(flowers_split_inds, datasets.Dataset.SPLIT_KEYS['validation'],
                flowers_splits['valid'][0].tolist())

    flowers_split_inds = sorted(flowers_split_inds, key=itemgetter(0))

    with open(flowers_dir / 'train_val_test_split.txt', 'w+') as f:
        for id_and_type in flowers_split_inds:
            f.write('%s %s\n' % id_and_type)

    # "paths"
    default_images_path = flowers_dir / 'jpg'
    images_path = flowers_dir / 'images'

    if not images_path.exists():
        default_images_path.rename(images_path)

    paths_path = flowers_dir / 'paths.txt'
    if not paths_path.exists():
        paths = [f.name.split('_')[1] for f in sorted(images_path.iterdir())]
        with open(paths_path, 'w+') as f:
            for (i, path) in enumerate(paths):
                f.write('%i %s\n' % (i + 1, path))

    default_labels_path = flowers_dir / 'segmim'
    labels_path = flowers_dir / 'labels'
    if not labels_path.exists():
        default_labels_path.rename(labels_path)

    for label in labels_path.iterdir():
        label.rename(str(label).replace('segmim', 'label'))


def configure_faces(datasets_dir):
    pass


def configure_birds(datasets_dir):
    birds_dir = datasets_dir / 'Birds'

    birds_dir.mkdir(exist_ok=True)

    get_file(
        fname='images.tgz',
        origin=(
            'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_'
            '2011.tgz'),
        cache_subdir=birds_dir.absolute(),
        extract=True
    )

    get_file(
        fname='segments.tgz',
        origin=('http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/'
                'segmentations.tgz'),
        cache_subdir=birds_dir.absolute(),
        extract=True
    )

    get_file(
        fname='train_val_test_split.txt',
        origin=('https://raw.githubusercontent.com/mickaelChen/ReDO/master/'
                'datasplits/cub/train_val_test_split.txt'),
        cache_subdir=birds_dir.absolute()
    )

    default_image_path = birds_dir / 'CUB_200_2011' / 'images'
    image_path = birds_dir / 'images'

    default_segmentations_path = birds_dir / 'segmentations'
    segmentations_path = birds_dir / 'labels'

    default_paths_path = birds_dir / 'CUB_200_2011' / 'images.txt'
    paths_path = birds_dir / 'paths.txt'

    if not image_path.exists():
        default_image_path.rename(image_path)
    if not segmentations_path.exists():
        default_segmentations_path.rename(segmentations_path)
    if not paths_path.exists():
        default_paths_path.rename(paths_path)


datasets_dir = Path('Datasets')
datasets_dir.mkdir(exist_ok=True)

download_args = parse_download_args()

if not download_args.flowers and \
        not download_args.faces and \
        not download_args.birds:
    configure_birds(datasets_dir)
    configure_faces(datasets_dir)
    configure_flowers(datasets_dir)
elif download_args.birds:
    configure_birds(datasets_dir)
elif download_args.faces:
    configure_faces(datasets_dir)
elif download_args.flowers:
    configure_flowers(datasets_dir)
