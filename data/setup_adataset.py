import tensorflow as tf
import numpy as np
import os
import glob, pandas

import pandas as pd
import pydicom as dicom
import skimage.transform, scipy.misc
import re
# to ignore deprecated warining in skimage.transform
import warnings;

warnings.filterwarnings('ignore')

DATA_PATH = '/data/SNUBH/Rotator/'
RAW_PATH = os.path.join(DATA_PATH, 'RAW')

IMAGE_SIZE = [512, 512]


def str_extract(string, text):

    extract = re.search(string, text)
    if extract is None:
        matching = None
    else:
        matching = extract.group()
    return matching


class Dataset_npy():
    def __init__(self, data_dir, batch_size, only_test, **kwargs):
        if not os.path.exists(data_dir):  # if pre-built dataset is available, kwarg is not required
            if 'clv_train' in kwargs:
                train_X1, train_Y1 = kwargs['file1_train'], kwargs['label1_train']
                train_X3, train_Y3 = kwargs['file3_train'], kwargs['label3_train']
                train_X4, train_Y4 = kwargs['file4_train'], kwargs['label4_train']
                train_Z, train_W = kwargs['id_train'], kwargs['clv_train']

            if 'clv_test' in kwargs:
                test_X1, test_Y1 = kwargs['file1_test'], kwargs['label1_test']
                test_X3, test_Y3 = kwargs['file3_test'], kwargs['label3_test']
                test_X4, test_Y4 = kwargs['file4_test'], kwargs['label4_test']
                test_Z, test_W = kwargs['id_test'], kwargs['clv_test']

            else:
                raise AssertionError('filenames or labels must be provided. please check npy file.')

            data_root = os.path.split(data_dir)[0]
            if not os.path.exists(data_root): os.makedirs(data_root)

            #import pdb; pdb.set_trace()

            if only_test:
                np.save(data_dir, {'test_X1': test_X1, 'test_Y1': test_Y1,
                                   'test_X3': test_X3, 'test_Y3': test_Y3,
                                   'test_X4': test_X4, 'test_Y4': test_Y4,
                                   'test_Z': test_Z, 'test_W': test_W
                                   })

            else:
                np.save(data_dir, {'train_X1': train_X1, 'train_Y1': train_Y1,
                                   'train_X3': train_X3, 'train_Y3': train_Y3,
                                   'train_X4': train_X4, 'train_Y4': train_Y4,
                                   'train_Z': train_Z, 'train_W': train_W,
                                   'test_X1': test_X1, 'test_Y1': test_Y1,
                                   'test_X3': test_X3, 'test_Y3': test_Y3,
                                   'test_X4': test_X4, 'test_Y4': test_Y4,
                                   'test_Z': test_Z, 'test_W': test_W
                                   })

        else:
            pre_built = np.load(data_dir).item()

            if only_test:
                test_X1, test_Y1 = pre_built['test_X1'], pre_built['test_Y1']
                test_X3, test_Y3 = pre_built['test_X3'], pre_built['test_Y3']
                test_X4, test_Y4 = pre_built['test_X4'], pre_built['test_Y4']
                test_Z, test_W = pre_built['test_Z'], pre_built['test_W']
                self.data_length = len(test_Z)

            else:
                train_X1, train_Y1 = pre_built['train_X1'], pre_built['train_Y1']
                train_X3, train_Y3 = pre_built['train_X3'], pre_built['train_Y3']
                train_X4, train_Y4 = pre_built['train_X4'], pre_built['train_Y4']
                train_Z, train_W = pre_built['train_Z'], pre_built['train_W']

                test_X1, test_Y1 = pre_built['test_X1'], pre_built['test_Y1']
                test_X3, test_Y3 = pre_built['test_X3'], pre_built['test_Y3']
                test_X4, test_Y4 = pre_built['test_X4'], pre_built['test_Y4']
                test_Z, test_W = pre_built['test_Z'], pre_built['test_W']

                self.data_length = len(train_Z) + len(test_Z)

        if only_test:
            self.test = self.Sub_Dataset((test_X1, test_Y1, test_X3, test_Y3, test_X4, test_Y4,
                                          test_Z, test_W),
                                         batch_size=batch_size, shuffle=False)
            self.id_test = test_Z

        else:
            # replicate positive images for training set
            index = np.asarray([v[-1] for v in train_Y1])

            train_X1 = np.concatenate([train_X1, train_X1[index == 1]], axis=0)
            train_X3 = np.concatenate([train_X3, train_X3[index == 1]], axis=0)
            train_X4 = np.concatenate([train_X4, train_X4[index == 1]], axis=0)
            train_Y1 = np.concatenate([train_Y1, train_Y1[index == 1]], axis=0)
            train_Y3 = np.concatenate([train_Y3, train_Y3[index == 1]], axis=0)
            train_Y4 = np.concatenate([train_Y4, train_Y4[index == 1]], axis=0)

            train_Z = np.concatenate([train_Z, train_Z[index == 1]], axis=0)
            train_W = np.concatenate([train_W, train_W[index == 1]], axis=0)

            np.random.seed(20190319)
            p = np.random.permutation(len(train_X1))
            train_X1, train_Y1 = train_X1[p], train_Y1[p]
            train_X3, train_Y3 = train_X3[p], train_Y3[p]
            train_X4, train_Y4 = train_X4[p], train_Y4[p]
            train_Z, train_W = train_Z[p], train_W[p]

            #train_X, train_Y = train_X[p], train_Y[p]
            self.train = self.Sub_Dataset((train_X1, train_Y1, train_X3, train_Y3, train_X4, train_Y4,
                                          train_Z, train_W),
                                         batch_size=batch_size, shuffle=True)

            self.test = self.Sub_Dataset((test_X1, test_Y1, test_X3, test_Y3, test_X4, test_Y4,
                                          test_Z, test_W),
                                         batch_size=batch_size, shuffle=False)

            self.id_train, self.id_test = train_Z, test_Z

    class Sub_Dataset():
        def __init__(self, filenames_n_labels, num_epochs=1, batch_size=1,
                     shuffle=False, augmentation=False):
            self.file1, self.label1, self.file2, self.label2, self.file3, self.label3,\
            self.id, self.clv = \
                filenames_n_labels
            self.data_length = len(self.clv)

            self.neg_length = len([v[-1] for v in self.label1 if v[-1] == 0])
            self.pos_length = self.data_length - self.neg_length

            dataset = tf.data.Dataset.from_tensor_slices(tensors=
                                                         (self.file1, [v for v in self.label1],
                                                          self.file2, [v for v in self.label2],
                                                          self.file3, [v for v in self.label3],
                                                          [v for v in self.id],
                                                          [v for v in self.clv]
                                                          ))
            if shuffle:
                dataset = dataset.shuffle(buffer_size=batch_size*100, reshuffle_each_iteration=True)

            def dicom_read_by_ftn(file1, label1, file2, label2, file3, label3, id, clv):
                def each_read(filename, label):
                    dicom_info = dicom.read_file(filename.decode())
                    x1, y1, x2, y2 = label[:-1]

                    image = dicom_info.pixel_array
                    # recover image if inverted
                    if str(dicom_info[0x28, 0x04].value) == 'MONOCHROME1':
                        white_image = np.full_like(image, np.max(image), image.dtype)
                        image = np.subtract(white_image, image)

                    # crop and resize
                    image = image[max(0, y1):min(dicom_info.Rows, y2),
                            max(0, x1):min(dicom_info.Columns, x2)]

                    image = np.expand_dims(skimage.transform.resize(image, IMAGE_SIZE, preserve_range=True),
                                           axis=-1)

                    image = (image - np.mean(image)) / np.std(image)  # normalization

                    return image.astype(np.float32), np.int64(label[-1])

                clv_split = re.split('_', clv.decode())
                clv_split = [np.float32(i) for i in clv_split]

                age, sxf, sxm, vas, tm0, tm1, tm2, dm0, dm1, dm2 = clv_split[:]

                if vas == 99.0:
                    vas = np.float32(6.0)

                f1, l1 = each_read(file1, label1)
                f2, l2 = each_read(file2, label2)
                f3, l3 = each_read(file3, label3)

                id = id.decode()

                return f1, f2, f3, l1, id, age, sxf, sxm, vas, tm0, tm1, tm2, dm0, dm1, dm2

            dataset = dataset.map(num_parallel_calls=8,
                                  map_func=lambda file1, label1, file2, label2, file3, label3, id, clv:
                                  tuple(tf.py_func(func=dicom_read_by_ftn,
                                                   inp=[file1, label1,
                                                        file2, label2,
                                                        file3, label3,
                                                        id, clv],
                                                   Tout=[tf.float32, tf.float32, tf.float32,
                                                         tf.int64, tf.string, tf.float32,
                                                         tf.float32, tf.float32, tf.float32,
                                                         tf.float32, tf.float32, tf.float32,
                                                         tf.float32, tf.float32, tf.float32
                                                         ])))

            if num_epochs == 0:
                dataset = dataset.repeat(count=num_epochs)  # raise out-of-range error when num_epochs done
                dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
            else:
                dataset = dataset.batch(batch_size)
                dataset = dataset.repeat(count=num_epochs)
            iterator = dataset.make_initializable_iterator()

            self.init_op = iterator.initializer
            self.next_batch = iterator.get_next()


def dataset_aggre(npy_name):
    npy_path = os.path.join(DATA_PATH, 'ro_new', config.exp_name, 'data')
    data_dir = os.path.join(npy_path, npy_name)
    xlsx_path = os.path.join(DATA_PATH, 'ro_new', config.exp_name, 'xlsx')
    if not os.path.exists(xlsx_path):
        os.makedirs(xlsx_path)

    xlsx = pd.DataFrame()

    xlsx_test = pd.read_excel(config.test_xlsx)
    xlsx_test['DATA_TYPE'] = 'test'
    xlsx = xlsx.append(xlsx_test, ignore_index=True)

    if config.train_xlsx != config.test_xlsx:
        xlsx_train = pd.read_excel(config.train_xlsx)
        xlsx_train['DATA_TYPE'] = 'train'
        xlsx = xlsx.append(xlsx_train, ignore_index=True)

    select_col = ['NUMBER', 'FOLDER_NAME', 'SIDE', 'LABEL_SST_BIN0',
                  'VIEW1', 'COORDS1_X', 'COORDS1_Y', 'SPACING1_X', 'SPACING1_Y',
                  'VIEW3', 'COORDS3_X', 'COORDS3_Y', 'SPACING3_X', 'SPACING3_Y',
                  'VIEW4', 'COORDS4_X', 'COORDS4_Y', 'SPACING4_X', 'SPACING4_Y',
                  'PATIENT_AGE', 'F', 'M', 'VAS_MED',
                  'TRAUMA0', 'TRAUMA1', 'TRAUMA2',
                  'DOMINANT0', 'DOMINANT1', 'DOMINANT2',
                  'DATA_TYPE'
                  ]

    xlsx = xlsx[select_col]

    def calculate_crop_coords(center_x, center_y, spacing_x, spacing_y, radius):
        x1, y1 = int(center_x - radius / spacing_x), int(center_y - radius / spacing_y)
        x2, y2 = int(center_x + radius / spacing_x), int(center_y + radius / spacing_y)
        return [x1, y1, x2, y2]

    def add_value(xlsx, view_type):
        view_name = 'VIEW'+view_type
        coord_x, coord_y = 'COORDS' + view_type + '_X', 'COORDS' + view_type + '_Y'
        spacing_x, spacing_y = 'SPACING' + view_type + '_X', 'SPACING' + view_type + '_Y'
        xlsx['FILENAMES'+view_type] = xlsx.apply(
            lambda row: os.path.join(RAW_PATH, row['FOLDER_NAME'], row[view_name]), axis=1)
        xlsx[view_name + '_COORDS'] = xlsx.apply(
            lambda row: calculate_crop_coords(center_x=row[coord_x], center_y=row[coord_y],
                                              spacing_x=row[spacing_x], spacing_y=row[spacing_y],
                                              radius=50), axis=1)
        xlsx['LABELS'+view_type] = xlsx.apply(lambda row: row[view_name+'_COORDS']+[row['LABEL_SST_BIN0']],
                                              axis=1)
        return xlsx

    def combine_clinical(patient_age, f, m, vas_med, trauma0, trauma1, trauma2,
                         dominant0, dominant1, dominant2):
        return '_'.join([str(patient_age), str(f), str(m), str(vas_med),
                         str(trauma0), str(trauma1), str(trauma2),
                         str(dominant0), str(dominant1), str(dominant2)
                          ])

    def files_labels_view(xlsx, view_type, data_type):
        filenames_view = xlsx[xlsx['DATA_TYPE'] == data_type]['FILENAMES'+view_type].values
        labels_view = xlsx[xlsx['DATA_TYPE'] == data_type]['LABELS' + view_type].values
        return filenames_view, labels_view

    xlsx = add_value(xlsx, '1')
    xlsx = add_value(xlsx, '3')
    xlsx = add_value(xlsx, '4')

    xlsx['CLINICAL'] = xlsx.apply(
        lambda row: combine_clinical(row['PATIENT_AGE'], row['F'], row['M'], row['VAS_MED'],
                             row['TRAUMA0'], row['TRAUMA1'], row['TRAUMA2'],
                             row['DOMINANT0'], row['DOMINANT1'], row['DOMINANT2']), axis=1)

    file1_test, label1_test = files_labels_view(xlsx, '1', 'test')
    file3_test, label3_test = files_labels_view(xlsx, '3', 'test')
    file4_test, label4_test = files_labels_view(xlsx, '4', 'test')

    id_test = xlsx[xlsx['DATA_TYPE'] == 'test']['NUMBER'].values
    clv_test =xlsx[xlsx['DATA_TYPE'] == 'test']['CLINICAL'].values
    test_info = pd.DataFrame({'FILENAMES1': pd.Series(file1_test),
                              'FILENAMES3': pd.Series(file3_test),
                              'FILENAMES4': pd.Series(file4_test),
                              'LABELS1': pd.Series(label1_test),
                              'LABELS3': pd.Series(label3_test),
                              'LABELS4': pd.Series(label4_test),
                              'CLINICAL': pd.Series(clv_test),
                              'ID': pd.Series(id_test)})

    test_info.to_csv(os.path.join(xlsx_path, npy_name + '_test.csv'))

    only_test = config.train_xlsx == config.test_xlsx

    print('only test: ', only_test)

    if only_test:
        dataset = Dataset_npy(data_dir=data_dir, batch_size=9, only_test=only_test,
                              file1_test=file1_test, label1_test=label1_test,
                              file3_test=file3_test, label3_test=label3_test,
                              file4_test=file4_test, label4_test=label4_test,
                              id_test=id_test, clv_test=clv_test)

    else:

        file1_train, label1_train = files_labels_view(xlsx, '1', 'train')
        file3_train, label3_train = files_labels_view(xlsx, '3', 'train')
        file4_train, label4_train = files_labels_view(xlsx, '4', 'train')

        id_train = xlsx[xlsx['DATA_TYPE'] == 'train']['NUMBER'].values
        clv_train = xlsx[xlsx['DATA_TYPE'] == 'train']['CLINICAL'].values
        train_info = pd.DataFrame({'FILENAMES1': pd.Series(file1_train),
                                  'FILENAMES3': pd.Series(file3_train),
                                  'FILENAMES4': pd.Series(file4_train),
                                  'LABELS1': pd.Series(label1_train),
                                  'LABELS3': pd.Series(label3_train),
                                  'LABELS4': pd.Series(label4_train),
                                  'CLINICAL': pd.Series(clv_train),
                                  'ID': pd.Series(id_train)})

        train_info.to_csv(os.path.join(xlsx_path, npy_name + '_train.csv'))

        dataset = Dataset_npy(data_dir=data_dir, batch_size=9, only_test=only_test,
                              file1_test=file1_test, label1_test=label1_test,
                              file3_test=file3_test, label3_test=label3_test,
                              file4_test=file4_test, label4_test=label4_test,
                              id_test=id_test, clv_test=clv_test,
                              file1_train=file1_train, label1_train=label1_train,
                              file3_train=file3_train, label3_train=label3_train,
                              file4_train=file4_train, label4_train=label4_train,
                              id_train=id_train, clv_train=clv_train
                              )

        #import pdb; pdb.set_trace()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    setup_config = parser.add_argument_group('dataset setting')
    setup_config.add_argument('--exp_name', type=str, dest='exp_name', default='exp309',
                              help='experiment name')
    setup_config.add_argument('--npy_name', type=str, dest='npy_name', default='exp309_test_all',
                              help='npy name')
    setup_config.add_argument('--train_xlsx', type=str, dest='train_xlsx',
                              default=DATA_PATH+'exp308/xlsx/exp308_test.xlsx')
    setup_config.add_argument('--test_xlsx', type=str, dest='test_xlsx',
                              default=DATA_PATH+'exp308/xlsx/exp308_test.xlsx')

    parser.print_help()
    config, unparsed = parser.parse_known_args()

    dataset_aggre(npy_name=config.npy_name)

