from redo.tests import test_datasets, test_discriminator, test_generator, \
    test_instance_norm, test_segmentation_network

test_datasets.test()
test_discriminator.test()
test_generator.test()
test_instance_norm.test()
test_segmentation_network.test()