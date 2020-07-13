from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('image_size', 248,
                            """input image-size of this network""")
tf.app.flags.DEFINE_integer('preprocess_size', 256,
                            """input image-size of this network""")

REMOTE_DATA_ROOT = 'REMOTE_DATA_ROOT'

class TFRecordRandomDataset(tf.data.Dataset):
    def __init__(self, input_dataset, filename_prefix, buffer_output_elements, num_threads):
        super(TFRecordRandomDataset, self).__init__()
        self._input_dataset = input_dataset
        self._filename_prefix = tf.convert_to_tensor(filename_prefix)
        self._compression_type = tf.convert_to_tensor('')
        self._buffer_output_elements = tf.convert_to_tensor(buffer_output_elements)
        self._num_threads = tf.convert_to_tensor(num_threads)

    def _as_variant_tensor(self):
        return _ops.tf_record_random_dataset(
            self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
            filename_prefix=self._filename_prefix,
            compression_type=self._compression_type,
            buffer_output_elements=self._buffer_output_elements,
            num_threads=self._num_threads)

    @property
    def output_classes(self):
        #return tf.Tensor
        return tuple([tf.Tensor] + list(self._input_dataset.output_classes[2:]))

    @property
    def output_shapes(self):
        #return tf.TensorShape([])
        return tuple([tf.TensorShape([])] + list(self._input_dataset.output_shapes[2:]))

    @property
    def output_types(self):
        #return tf.string
        return tuple([tf.string] + list(self._input_dataset.output_types[2:]))

class TFRecordParser(object):
    """TFRecord Example Parser
    """
    def __init__(self, pattern):
        """Create an `TFRecordParser` object.

        Args:
            pattern (``list`` of (``name``, ``raw_type``, ``target_type``, ``shape``)): 
                A list of tuples indicating how to parse the tfrecord. 
                ``name`` is feature keys. ``shape`` is shape of the feature. If ``raw_type`` is tf.string and different from ``target_type``, 
                the bytes will be decoded to ``target_type``.
        """
        # [(name, raw_type, type_after_decode, shape) ...]
        self._pattern = pattern
        self._sample_processor = None

    def __call__(self, sample_proto, *others):
        pattern = {}
        for name, raw_t, t, shape in self._pattern:
            if shape is None:
                pattern[name] = tf.VarLenFeature(raw_t)
            else:
                pattern[name] = tf.FixedLenFeature(shape, raw_t)
        features = tf.parse_single_example(sample_proto, pattern)

        outs = []
        for name, raw_t, t, shape in self._pattern:
            out = features[name]
            if raw_t == tf.string and t != raw_t:
                out = tf.decode_raw(out, t)
            outs.append(out)
        outs.extend(others)
        if self._sample_processor is not None:
            outs = self._sample_processor(*outs)

        return tuple(outs)


class SampleProcessor(object):
    def __init__(self, flip):
        self._resize_size = FLAGS.preprocess_size
        self._crop_size = FLAGS.image_size
        self.flip = flip
        #self._resize_size = 320
        #self._crop_size = 316
    
    def __call__(self, file_name):
        def _resize_preserving_aspect(image, new_shorter_edge):
          img_shape = tf.shape(image)
          height = img_shape[0]
          width = img_shape[1]

          edge = tf.convert_to_tensor(new_shorter_edge, dtype=tf.int32)

          def _calc_longer(long_edge, short_edge, new_short):
              return tf.cast(new_short * long_edge / short_edge, tf.int32)

          new_size = tf.cond(tf.less_equal(height, width),
                             lambda: (edge, _calc_longer(width, height, edge)),
                             lambda: (_calc_longer(height, width, edge), edge))
          return tf.image.resize_images(image, new_size)
        
        file_reader = tf.read_file(file_name)
        #img_reader = tf.image.decode_jpeg(file_reader, channels=3)
        img_reader = tf.image.decode_image(file_reader, channels=3)
        image = tf.image.convert_image_dtype(img_reader, dtype=tf.float32)
        if self.flip:
            image = tf.image.flip_left_right(image)
        image.set_shape([self._resize_size, self._resize_size, 3])
        image = _resize_preserving_aspect(image, self._resize_size)
        image = tf.image.resize_image_with_crop_or_pad(image, self._crop_size, self._crop_size)
        return (file_name, image)

class TFRecordSampleProcessor(object):
    def __init__(self):
        self._resize_size = 256
        self._crop_size = 248
    
    def __call__(self, image_buffer, file_name):
        def _resize_preserving_aspect(image, new_shorter_edge):
          img_shape = tf.shape(image)
          height = img_shape[0]
          width = img_shape[1]

          edge = tf.convert_to_tensor(new_shorter_edge, dtype=tf.int32)

          def _calc_longer(long_edge, short_edge, new_short):
              return tf.cast(new_short * long_edge / short_edge, tf.int32)

          new_size = tf.cond(tf.less_equal(height, width),
                             lambda: (edge, _calc_longer(width, height, edge)),
                             lambda: (_calc_longer(height, width, edge), edge))
          return tf.image.resize_images(image, new_size)
        
        #file_name = tf.Print(file_name, [file_name], message='FILE NAME: ')
        img_reader = tf.image.decode_jpeg(image_buffer, channels=3)
        image = tf.image.convert_image_dtype(img_reader, dtype=tf.float32)

        image = _resize_preserving_aspect(image, self._resize_size)
        image = tf.image.resize_image_with_crop_or_pad(image, self._crop_size, self._crop_size)
        return (file_name, image)


class LineParser(object):
    def __init__(self):
        self._sample_num = 0
        self._broken_num = 0

    def __call__(self, line):
        if '\t' in line:
            file_name = line.strip().split('\t')[0]
        else:
            file_name = line.strip().split(' ')[0]

        if not tf.gfile.Exists(file_name):
            self._broken_num += 1
            raise ValueError('Faild to find file: ' + file_name)
        self._sample_num += 1
        return (file_name, ) 


class TFRecordIndexParser(object):
    def __init__(self):
        self._sample_num = 0
        self._broken_num = 0

    def __call__(self, line):
        line = line.strip()
        file_name, shard_index, shard_offset = line.split('\t')
        return ['{0}/{0}-{1:05}'.format('TFR-ECCV_fairface_train_256', int(shard_index)), int(shard_offset), file_name]


class BaseData(object):
    def __init__(self, sample_processor, name=None):
        self._sample_processor = sample_processor
        self._name = name
        self._dataset = None
        self._iterator = None
        self._batch_size = None
    
    def build(self):
        self._dataset = self._make_dataset()

    def _make_dataset(self):
        raise NotImplementedError

    def get_iterator(self):
        return self._iterator

    def base_batch_input(self, batch_size,
                    epoch_num=1,
                    process_parallel_num=1,
                    prefetch_buffer_size=1,
                    filter_after_shuffle_repeat_prefetch=None,
                    sample_parser=None,
                    ):
        ds = self._dataset
        ds = ds.prefetch(prefetch_buffer_size)
        ds = ds.repeat()

        if filter_after_shuffle_repeat_prefetch is not None:
            ds = filter_after_shuffle_repeat_prefetch(ds)
        
        if sample_parser is not None: 
            ds = ds.map(map_func=sample_parser,
                       num_parallel_calls=process_parallel_num)
            ds = ds.prefetch(buffer_size=prefetch_buffer_size)
        
        if self._sample_processor is not None:
            ds = ds.apply(
                   tf.contrib.data.map_and_batch(
                        map_func=self._sample_processor,
                        batch_size=batch_size,
                        num_parallel_batches=process_parallel_num,
                        drop_remainder=True))
            ds = ds.prefetch(prefetch_buffer_size)
        else:
            ds = ds.batch(batch_size=batch_size)
        
        self._iterator = ds.make_initializable_iterator()
        self._batch_size = batch_size
        return self._iterator.get_next()

        
class TarData(BaseData):
    def __init__(self, name, index_file, index_parser, sample_processor):
        super(TarData, self).__init__(sample_processor, name)
        self._index_file = index_file
        self._index_parser = index_parser    
        self._inputs = None
        self._placeholders = None
        self._sample_num = 0
    
    def epoch_size(self):
        return self._sample_num

    def _dtype_np2tf(self, d): 
        t = { 
            np.bool: tf.bool,
            np.int8: tf.int8,
            np.int16: tf.int16,
            np.int32: tf.int32,
            np.int64: tf.int32,
            np.float16: tf.float16,
            np.float32: tf.float32,
            np.float64: tf.float64,
        }   
        for k, v in t.items():
            if k == d:
                return v
        return tf.string

    def _make_dataset(self):
        self.inputs = []
        with open(self._index_file, 'r') as f:
            for line_i, line in enumerate(f):
                #print(line)
                inputs = self._index_parser(line)
                if inputs is None:
                    continue
                self._sample_num += 1
                if self._inputs is None:
                    self._inputs = [[] for i in range(len(inputs))]
                for i, x in enumerate(inputs):
                    self._inputs[i].append(x)
        
        self._inputs = [np.array(x) for x in self._inputs]
        self._placeholders = [tf.placeholder(self._dtype_np2tf(x.dtype), x.shape) for x in self._inputs]
        return tf.data.Dataset.from_tensor_slices(tuple(self._placeholders))

    def batch_input(self, 
                    batch_size, 
                    epoch_num=1,
                    process_parallel_num=1,
                    prefetch_buffer_size=1):

        return self.base_batch_input(batch_size, 
                                    epoch_num, 
                                    process_parallel_num,
                                    prefetch_buffer_size, 
                                    filter_after_shuffle_repeat_prefetch=None,
                                    sample_parser=None)

    def init_iterator(self, sess):
        sess.run(self._iterator.initializer, feed_dict={x:y for x, y in zip(self._placeholders, self._inputs)})

class TFRecordData(BaseData):
    def __init__(self, name, index_file, index_parser, sample_processor):
        super(TFRecordData, self).__init__(sample_processor, name)
        self._index_file = index_file
        self._prefix = REMOTE_DATA_ROOT
        self._index_parser = index_parser    
        self._inputs = None
        self._placeholders = None
        self._sample_num = 0
        self._pattern = [('image', tf.string, tf.string, ())]
        
    def epoch_size(self):
        return self._sample_num

    def _dtype_np2tf(self, d): 
        t = { 
            np.bool: tf.bool,
            np.int8: tf.int8,
            np.int16: tf.int16,
            np.int32: tf.int32,
            np.int64: tf.int32,
            np.float16: tf.float16,
            np.float32: tf.float32,
            np.float64: tf.float64,
        }   
        for k, v in t.items():
            if k == d:
                return v
        return tf.string

    def _make_dataset(self):
        self.inputs = []
        with open(self._index_file, 'r') as f:
            for line_i, line in enumerate(f):
                inputs = self._index_parser(line)
                if inputs is None:
                    continue
                self._sample_num += 1
                if self._inputs is None:
                    self._inputs = [[] for i in range(len(inputs))]
                for i, x in enumerate(inputs):
                    self._inputs[i].append(x)
        
        self._inputs = [np.array(x) for x in self._inputs]
        dtypes = [tf.string, tf.int64] + [self._dtype_np2tf(x.dtype) for x in self._inputs[2:]]
        self._placeholders = [tf.placeholder(dtype, x.shape) for dtype, x in zip(dtypes, self._inputs)]
        return tf.data.Dataset.from_tensor_slices(tuple(self._placeholders))
    
    def after_shuffle_repeat_prefetch(self, ds):
        return TFRecordRandomDataset(ds, self._prefix, buffer_output_elements=256, num_threads=8)
    

    def batch_input(self, 
                    batch_size, 
                    epoch_num=1,
                    process_parallel_num=1,
                    prefetch_buffer_size=1,
                    ):
       
        return self.base_batch_input(batch_size, 
                                    epoch_num, 
                                    process_parallel_num,
                                    prefetch_buffer_size, 
                                    filter_after_shuffle_repeat_prefetch=self.after_shuffle_repeat_prefetch,
                                    sample_parser=TFRecordParser(self._pattern))
        
    def init_iterator(self, sess):
        sess.run(self._iterator.initializer, feed_dict={x:y for x, y in zip(self._placeholders, self._inputs)})

