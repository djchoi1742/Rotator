import tensorflow as tf
import numpy as np
import os
import glob, pandas

import pydicom as dicom
import skimage.transform, scipy.misc
import re
# to ignore deprecated warining in skimage.transform
import warnings;

warnings.filterwarnings('ignore')

DATA_PATH = '/data/SNUBH/Rotator/'
# RAW_PATH = os.path.join(DATA_PATH, 'RAW')
RAW_PATH = os.path.join(DATA_PATH, 'SST_SELECT')

IMG_SIZE = 512

IMAGE_SIZE = [IMG_SIZE, IMG_SIZE]


def str_extract(string, text):
    extract = re.search(string, text)
    if extract is None:
        matching = None
    else:
        matching = extract.group()
    return matching


class Dataset_v2():
    def __init__(self, data_dir, batch_size, **kwargs):
        if not os.path.exists(data_dir):  # if pre-built dataset is available, kwarg is not required
            if 'filenames_train' in kwargs and 'filenames_test' in kwargs:
                filenames_train, filenames_test = kwargs['filenames_train'], kwargs['filenames_test']
            if 'labels_train' in kwargs and 'labels_test' in kwargs:
                labels_train, labels_test = kwargs['labels_train'], kwargs['labels_test']
            if 'id_train' in kwargs and 'id_test' in kwargs:
                id_train, id_test = kwargs['id_train'], kwargs['id_test']
            else:
                raise AssertionError('filenames or labels must be provided. please check npy file.')

            train_X, train_Y, train_Z = filenames_train, labels_train, id_train
            test_X, test_Y, test_Z = filenames_test, labels_test, id_test

            data_root = os.path.split(data_dir)[0]
            if not os.path.exists(data_root): os.makedirs(data_root)
            np.save(data_dir, {'train_X': train_X, 'train_Y': train_Y, 'test_X': test_X, 'test_Y': test_Y,
                               'train_Z': train_Z, 'test_Z': test_Z})

        else:
            pre_built = np.load(data_dir).item()
            train_X, train_Y, train_Z = pre_built['train_X'], pre_built['train_Y'], pre_built['train_Z']
            test_X, test_Y, test_Z = pre_built['test_X'], pre_built['test_Y'], pre_built['test_Z']

            self.data_length = len(train_X) + len(test_X)

        # replicate positive images for training set
        index = np.asarray([v[-1] for v in train_Y])
        train_X = np.concatenate([train_X, train_X[index == 1]], axis=0)
        train_Y = np.concatenate([train_Y, train_Y[index == 1]], axis=0)

        p = np.random.permutation(len(train_X))
        train_X, train_Y = train_X[p], train_Y[p]

        self.train = self.Sub_Dataset((train_X, train_Y), batch_size=batch_size, shuffle=True,
                                      augmentation=True)
        self.test = self.Sub_Dataset((test_X, test_Y), batch_size=batch_size, shuffle=False)

        self.id_train, self.id_test = train_Z, test_Z

    class Sub_Dataset():
        def __init__(self, filenames_n_labels, num_epochs=1, batch_size=1, shuffle=False, augmentation=False):
            self.filenames, self.labels = filenames_n_labels
            self.data_length = len(self.filenames)

            self.neg_length = len([v[-1] for v in self.labels if v[-1] == 0])
            self.pos_length = self.data_length - self.neg_length

            dataset = tf.data.Dataset.from_tensor_slices(tensors=(self.filenames, [v for v in self.labels]))

            if shuffle == True:
                dataset.shuffle(buffer_size=len(self.filenames), reshuffle_each_iteration=True)

            def _dicom_read_py_function(filename, label, augmentation):
                dicom_info = dicom.read_file(filename.decode())
                x1, y1, x2, y2 = label[:-1]

                if augmentation:  # maximum 10% variation for shifting
                    shift_x = np.random.randint(dicom_info.Columns // 10)
                    shift_y = np.random.randint(dicom_info.Rows // 10)

                    # shifting direction
                    shift_x = -shift_x if np.random.rand() <= 0.5 else shift_x
                    shift_y = -shift_y if np.random.rand() <= 0.5 else shift_y

                    x1 = x1 - shift_x
                    y1 = y1 - shift_y
                    x2 = x2 - shift_x
                    y2 = y2 - shift_y

                image = dicom_info.pixel_array
                # recover image if inverted
                if str(dicom_info[0x28, 0x04].value) == 'MONOCHROME1':
                    white_image = np.full_like(image, np.max(image), image.dtype)
                    image = np.subtract(white_image, image)

                # crop and resize
                image = image[max(0, y1):min(dicom_info.Rows, y2), max(0, x1):min(dicom_info.Columns, x2)]
                image = np.expand_dims(skimage.transform.resize(image, IMAGE_SIZE, preserve_range=True),
                                       axis=-1)

                if augmentation and np.random.randint(2) == 1:
                    image = np.fliplr(image)

                image = (image - np.mean(image)) / np.std(image)  # normalization
                return image.astype(np.float32), np.int64(label[-1])

            dataset = dataset.map(num_parallel_calls=8, map_func=lambda filename, label:
            tuple(tf.py_func(func=_dicom_read_py_function, inp=[filename, label, augmentation],
                             Tout=[tf.float32, tf.int64])))

            if num_epochs == 0:
                dataset = dataset.repeat(count=num_epochs)  # raise out-of-range error when num_epochs done
                dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
            else:
                dataset = dataset.batch(batch_size)
                dataset = dataset.repeat(count=num_epochs)
            iterator = dataset.make_initializable_iterator()

            self.init_op = iterator.initializer
            self.next_batch = iterator.get_next()


# pre-building training & validation set: python setup_dataset.py
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    setup_config = parser.add_argument_group('dataset setting')
    setup_config.add_argument('--view_type', type=int, dest='view_type', default=4,
                              help='view type: 1,3,4')
    setup_config.add_argument('--exp_name', type=str, dest='exp_name', default='exp309',
                              help='experiment name')
    setup_config.add_argument('--data_name', type=str, dest='data_name', default='SSC_SST',
                              help='dataset name')
    setup_config.add_argument('--train_xlsx', type=str, dest='train_xlsx',
                              default=DATA_PATH+'SST_info/ssc_sst_test22.xlsx',
                              help='train_xlsx')
    setup_config.add_argument('--test_xlsx', type=str, dest='test_xlsx',
                              default=DATA_PATH+'SST_info/ssc_sst_test22.xlsx',
                              help='test_xlsx')
    setup_config.add_argument('--data_prop', type=float, dest='data_prop', default=1.0, help='data_prop')

    parser.print_help()
    config, unparsed = parser.parse_known_args()

    view_type = str(config.view_type)
    exp_name = config.exp_name
    data_name = config.data_name

    view_name = 'VIEW'+view_type
    npy_path = os.path.join(DATA_PATH, 'ro_new', exp_name, 'data')
    xlsx_path = os.path.join(DATA_PATH, 'ro_new', exp_name, 'xlsx')

    if not os.path.exists(npy_path):
        os.makedirs(npy_path)
    if not os.path.exists(xlsx_path):
        os.makedirs(xlsx_path)

    npy_name = data_name + '_' + view_name + '.npy'

    coord_x = 'COORDS'+view_type+'_X'
    coord_y = 'COORDS'+view_type+'_Y'
    spacing_x, spacing_y = 'SPACING' + view_type + '_X', 'SPACING' + view_type + '_Y'

    data_dir = npy_path + '/' + npy_name

    if True:
    #if not os.path.exists(data_dir):
        import pandas as pd
        import glob
        reference_root = DATA_PATH+'exp286/xlsx'

        xlsx = pd.DataFrame()

        xlsx_train = pd.read_excel(config.train_xlsx)
        xlsx_train['DATA_TYPE'] = 'train'
        np.random.seed(20190408)
        xlsx_train['DATA_INDEX'] = np.random.permutation(np.arange(len(xlsx_train)))
        xlsx_train = xlsx_train[xlsx_train['DATA_INDEX'] < config.data_prop * len(xlsx_train)]
        xlsx = xlsx.append(xlsx_train, ignore_index=True)

        xlsx_test = pd.read_excel(config.test_xlsx)
        xlsx_test['DATA_TYPE'] = 'test'
        np.random.seed(20190409)
        xlsx_test['DATA_INDEX'] = np.random.permutation(np.arange(len(xlsx_test)))
        xlsx_test = xlsx_test[xlsx_test['DATA_INDEX'] < config.data_prop * len(xlsx_test)]
        xlsx = xlsx.append(xlsx_test, ignore_index=True)

        select_col = ['NUMBER', 'FOLDER_NAME', 'SIDE', 'LABEL_SST_BIN0', view_name,
                      coord_x, coord_y, spacing_x, spacing_y, 'DATA_TYPE', 'DATA_INDEX']
        xlsx = xlsx[select_col]

        def calculate_crop_coords(center_x, center_y, spacing_x, spacing_y, radius):
            x1, y1 = int(center_x - radius / spacing_x), int(center_y - radius / spacing_y)
            x2, y2 = int(center_x + radius / spacing_x), int(center_y + radius / spacing_y)
            return [x1, y1, x2, y2]

        xlsx['FILENAMES'] = xlsx.apply(
            lambda row: os.path.join(RAW_PATH, row['FOLDER_NAME'], row[view_name]), axis=1)
        xlsx[view_name+'_COORDS'] = xlsx.apply(
            lambda row: calculate_crop_coords(center_x=row[coord_x], center_y=row[coord_y],
                                              spacing_x=row[spacing_x], spacing_y=row[spacing_y], radius=50),
                                              axis=1)
        xlsx['LABELS'] = xlsx.apply(lambda row: row[view_name+'_COORDS'] + [row['LABEL_SST_BIN0']], axis=1)

        filenames_train = xlsx[xlsx['DATA_TYPE'] == 'train']['FILENAMES'].values
        filenames_test = xlsx[xlsx['DATA_TYPE'] == 'test']['FILENAMES'].values

        labels_train = xlsx[xlsx['DATA_TYPE'] == 'train']['LABELS'].values
        labels_test = xlsx[xlsx['DATA_TYPE'] == 'test']['LABELS'].values

        id_train = xlsx[xlsx['DATA_TYPE'] == 'train']['NUMBER'].values
        id_test = xlsx[xlsx['DATA_TYPE'] == 'test']['NUMBER'].values

        index_train = xlsx[xlsx['DATA_TYPE'] == 'train']['DATA_INDEX'].values
        index_test = xlsx[xlsx['DATA_TYPE'] == 'test']['DATA_INDEX'].values

        train_info = pd.DataFrame({'FILENAMES': pd.Series(filenames_train),
                                   'LABELS': pd.Series(labels_train),
                                   'ID': pd.Series(id_train),
                                   'INDEX': pd.Series(index_train)})

        test_info = pd.DataFrame({'FILENAMES': pd.Series(filenames_test),
                                  'LABELS': pd.Series(labels_test),
                                  'ID': pd.Series(id_test),
                                  'INDEX': pd.Series(index_test)})

        if config.train_xlsx != config.test_xlsx:
            train_info.to_csv(xlsx_path + '/' + view_name + '_' + data_name + '_' + 'train.csv')
        test_info.to_csv(xlsx_path + '/' + view_name + '_' + data_name + '_' + 'test.csv')

        dataset = Dataset_v2(data_dir=data_dir, batch_size=9, view_name=view_name,
                            filenames_train=filenames_train, filenames_test=filenames_test,
                            labels_train=labels_train, labels_test=labels_test,
                            id_train=id_train, id_test=id_test
                            #data_name=data_name
                            )  # kwargs for pre-building dataset

    #else:  # use pre-built dataset

    #    dataset = Dataset_v2(data_dir=data_dir, batch_size=9, view_name=view_name, data_name=data_name)

    print('# of train set:', dataset.train.data_length, '  # of negative train set:', dataset.train.neg_length,
          '  # of positive train set (replicated):', dataset.train.pos_length)
    print('# of test set:', dataset.test.data_length, '  # of negative test set:', dataset.test.neg_length,
          '  # of positive train set:', dataset.test.pos_length)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    #os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    def test_dataset_loader():
        sess = tf.Session()
        sess.run([dataset.train.init_op, dataset.test.init_op])

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        axs = [axs[row][col] for row in range(3) for col in range(3)]

        if not os.path.exists('feeding_examples'):
            os.mkdir('feeding_examples')

        count = 0
        while True:
            try:
                images, labels = sess.run(dataset.test.next_batch)
            except tf.errors.OutOfRangeError:
                break

            print(images.shape, labels.shape, labels)
            for i, image in enumerate(images):
                axs[i].imshow(image[:, :, 0], cmap=plt.cm.gray)

            plt.savefig('feeding_examples/{0}.png'.format(count));
            plt.cla()
            count += 1


    #import pdb;

    #pdb.set_trace()
    #test_dataset_loader()