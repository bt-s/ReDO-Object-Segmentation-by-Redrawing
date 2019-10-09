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
path assuming the common structure shown above. Though differences may appear in
each subclass of dataset.Dataset
"""

from tensorflow.keras.utils import get_file
from scipy.io import loadmat
from PIL import Image

from operator import itemgetter
from argparse import ArgumentParser, Action
from sys import argv
from pathlib import Path
from shutil import rmtree
from os.path import splitext

import datasets


def parse_download_args():
    parser = ArgumentParser()
    parser.add_argument('-f', '--flowers', action='append_const',
                        const='flowers', dest='datasets')
    parser.add_argument('-b', '--birds', action='append_const', const='birds',
                        dest='datasets')
    parser.add_argument('-t', '--faces', action='append_const', const='faces',
                        dest='datasets')
    return parser.parse_args(argv[1:])


def _rm_dirs(dirs_to_rm):
    for a_dir in dirs_to_rm:
        if a_dir.exists():
            rmtree(a_dir)


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
                flowers_splits['tstid'][0].tolist())
    add_to_inds(flowers_split_inds, datasets.Dataset.SPLIT_KEYS['test'],
                flowers_splits['trnid'][0].tolist())
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

    _rm_dirs([default_images_path, default_labels_path])


def _read_face_split(fname):
    image_names_in_split = set()
    with open(fname, 'r') as reader:
        for line in reader.readlines():
            person, num = line.strip().split(' ')
            image_names_in_split.add('%s_%s.jpg' % (person, num.zfill(4)))
    return image_names_in_split


def _get_set_membership(train_set, test_set, valid_set, image_name):
    if image_name in train_set:
        return datasets.Dataset.SPLIT_KEYS['training']
    elif image_name in test_set:
        return datasets.Dataset.SPLIT_KEYS['test']
    elif image_name in valid_set:
        return datasets.Dataset.SPLIT_KEYS['validation']
    else:
        return None


def configure_faces(datasets_dir):
    faces_dir = datasets_dir / 'Faces'
    faces_dir.mkdir(exist_ok=True)

    get_file(
        fname='images.tgz',
        origin='http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz',
        cache_subdir=faces_dir.absolute(),
        extract=True,
        file_hash='1b42dfed7d15c9b2dd63d5e5840c86ad'
    )

    get_file(
        fname='labels.tgz',
        origin=('http://vis-www.cs.umass.edu/lfw/part_labels'
                '/parts_lfw_funneled_gt_images.tgz'),
        cache_subdir=faces_dir.absolute(),
        extract=True,
        file_hash='3e7e26e801c3081d651c8c2ef3c45cfc'
    )

    train_file = get_file(
        fname='train.txt',
        origin='http://vis-www.cs.umass.edu/lfw/part_labels/parts_train.txt',
        cache_subdir=faces_dir.absolute()
    )

    test_file = get_file(
        fname='test.txt',
        origin='http://vis-www.cs.umass.edu/lfw/part_labels/parts_test.txt',
        cache_subdir=faces_dir.absolute()
    )

    valid_file = get_file(
        fname='valid.txt',
        origin=('http://vis-www.cs.umass.edu/lfw/part_labels/parts_validation'
                '.txt'),
        cache_subdir=faces_dir.absolute()
    )

    default_images_path = faces_dir / 'lfw_funneled'
    images_path = faces_dir / 'images'

    default_labels_path = faces_dir / 'parts_lfw_funneled_gt_images'
    labels_path = faces_dir / 'labels'

    if not images_path.exists():
        default_images_path.rename(images_path)

    labels_path.mkdir(exist_ok=True)
    if default_labels_path.exists():
        for ppm_label in default_labels_path.iterdir():
            # Argh why are there these hidden files?
            if ppm_label.name[0] == '.':
                continue
            ppm_label_name, ppm_label_extension = splitext(ppm_label.name)
            Image.open(ppm_label).save(
                labels_path / ('%s.jpg' % ppm_label_name))

    paths_path = faces_dir / 'paths.txt'
    splits_path = faces_dir / 'train_val_test_split.txt'

    if not paths_path.exists():
        all_image_names = []
        for person_subdir in images_path.iterdir():
            if not person_subdir.is_dir():
                continue
            for image in person_subdir.iterdir():
                all_image_names.append(image.name)
        all_image_names = sorted(all_image_names)
        with open(paths_path, 'w+') as writer:
            for image_pair in enumerate(all_image_names):
                writer.write('%s %s\n' % image_pair)

        train_set = _read_face_split(train_file)
        test_set = _read_face_split(test_file)
        valid_set = _read_face_split(valid_file)

        with open(splits_path, 'w+') as writer:
            for (i, image_name) in enumerate(all_image_names):
                set_num = _get_set_membership(train_set, test_set, valid_set,
                                              image_name)
                if not set_num:
                    continue
                writer.write('%d %d\n' % (i, set_num))

    _rm_dirs([default_labels_path, default_images_path])


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

    default_extract_dir = birds_dir / 'CUB_200_2011'
    default_image_path = default_extract_dir / 'images'
    image_path = birds_dir / 'images'

    default_segmentations_path = birds_dir / 'segmentations'
    segmentations_path = birds_dir / 'labels'

    default_paths_path = default_extract_dir / 'images.txt'
    paths_path = birds_dir / 'paths.txt'

    if not image_path.exists():
        default_image_path.rename(image_path)
    if not segmentations_path.exists():
        default_segmentations_path.rename(segmentations_path)
    if not paths_path.exists():
        default_paths_path.rename(paths_path)

    _rm_dirs([default_extract_dir, default_segmentations_path])


SUPPORTED_DATASETS = {'flowers': configure_flowers, 'birds': configure_birds}

root_dataset_dir = Path('Datasets')
root_dataset_dir.mkdir(exist_ok=True)

download_args = parse_download_args()

if not download_args.datasets:
    download_args.datasets = SUPPORTED_DATASETS.keys()

for ds_key in download_args.datasets:
    SUPPORTED_DATASETS[ds_key](root_dataset_dir)
