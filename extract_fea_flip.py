from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import os.path
import time

import numpy as np
import tensorflow as tf
from data_flip import SampleProcessor, LineParser, TarData
from data_flip import TFRecordSampleProcessor, TFRecordIndexParser, TFRecordData

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path', '/tmp/data_path',
                            """Where to load """)
tf.app.flags.DEFINE_string('pb_path', '/tmp/model_file',
                            """Where to load check point""")
tf.app.flags.DEFINE_string('output_path', '/tmp/out_file',
                            """output features to the file""")
tf.app.flags.DEFINE_integer('num_gpus', 4, 'how many gpus to use')
tf.app.flags.DEFINE_integer('batch_size', 32, 'how many gpus to use')

def feature(data_path, out_file, flip):
    with tf.Graph().as_default():
        #data = TFRecordData('data', data_path, TFRecordIndexParser(), TFRecordSampleProcessor())
        data = TarData('data', data_path, LineParser(), SampleProcessor(flip))
        data.build()
        batch_in_epoch = math.ceil(data.epoch_size() / FLAGS.batch_size)
        batch_input = data.batch_input(FLAGS.batch_size)
        print(batch_input)
        fns, images = batch_input
        image_splits = tf.split(images, FLAGS.num_gpus, 0)

        # loading pb
        graph_def = tf.GraphDef()
        pb_path = FLAGS.pb_path
        with open(pb_path, 'rb') as f:
            graph_def.ParseFromString(f.read())
        
        pb_input = 'graph_input_0:0'
        pb_outputs_lst = ['l2_normalize:0']

        fea_splits = []
        for i in range(FLAGS.num_gpus):
            scp_name = 'fea_ext_{0}'.format(i)
            with tf.device('/gpu:{0}'.format(i)), tf.name_scope(scp_name):
                input_tensor_map = {pb_input: image_splits[i]}

                outputs = tf.import_graph_def(graph_def,
                                              input_map=input_tensor_map,
                                              return_elements=pb_outputs_lst,
                                              name='')
                fea_vec = outputs[0]
                print(fea_vec)
                fea_splits.append(fea_vec)
        features = tf.concat(fea_splits,0)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        data.init_iterator(sess)
        with open(out_file, 'w') as out_f:
            for n, fea in _extract_fea(features, fns, batch_in_epoch, sess):
                _write_feature(out_f, n.decode() ,fea)

        print("{}: Extract feat done!".format(datetime.now()))

def _extract_fea(fea_ext, fn_tsr, n_loop, sess):
    print("{}: Start extract feat".format(datetime.now()))

    start_time = time.time()
    step = 0
    for i in range(n_loop):
        fn_out, fea_out = sess.run([fn_tsr, fea_ext])

        step += 1
        if step % 50 == 0:
            duration = time.time() - start_time

            print('{0}: {1} of {2} processed.'.format(
                    datetime.now(), step, n_loop),
                    flush=True)
            start_time = time.time()

        fn_list = list(fn_out)
        fea_list = list(fea_out)

        for fn, fea in zip(fn_list, fea_list):
            yield fn, fea

def _write_feature(out_obj, file_name, fea_vec):
    out_obj.write(file_name)
    for x in np.nditer(fea_vec, order='C'):
        out_obj.write(' {0:f}'.format(float(x)))
    out_obj.write('\n')

def _load_img_list(list_path):
    f_list = []
    with open(list_path, 'r') as f:
        for line in f:
            item = line.rstrip()
            f_list.append(item)

    return f_list

def main(argv=None):
    feature(FLAGS.data_path, FLAGS.output_path + '.noFlip', False)
    feature(FLAGS.data_path, FLAGS.output_path + '.Flip', True)
    feats = {}
    for line in open(FLAGS.output_path + '.noFlip'):
        line_s = line.rstrip().split(' ')
        feat = [float(x) for x in line_s[1:]]
        name = line_s[0]
        feat = np.array(feat).astype('float32')
        feat /= np.linalg.norm(feat)
        feats[name] = feat

    for line in open(FLAGS.output_path + '.Flip'):
        line_s = line.rstrip().split(' ')
        feat = [float(x) for x in line_s[1:]]
        name = line_s[0]
        feat = np.array(feat).astype('float32')
        feat /= np.linalg.norm(feat)
        feats[name] += feat

    f = open(FLAGS.output_path + '.mean', 'w')

    for name in feats:
        feat = [str(k) for k in feats[name]]
        f.write(name + ' ' + ' '.join(feat) + '\n')
    f.close()

if __name__ == '__main__':
    tf.app.run()

