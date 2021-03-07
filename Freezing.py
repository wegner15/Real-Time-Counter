import tensorflow as tf
import sys
from tensorflow.python.platform import gfile

from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

with tf.compat.v1.Session() as sess:
    model_filename =r"C:\Users\Bogingo\PycharmProjects\imageClassification\MyCNN\saved_model.pb"
    with gfile.FastGFile(model_filename, 'rb') as f:
        data = compat.as_bytes(f.read())
        sm = saved_model_pb2.SavedModel()
        sm.ParseFromString(data)
        #print(sm)
        if 1 != len(sm.meta_graphs):
            print('More than one graph found. Not sure which to write')
            sys.exit(1)

      	#graph_def = tf.GraphDef()
        #graph_def.ParseFromString(sm.meta_graphs[0])
        g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)
LOGDIR=r'C:\Users\Bogingo\PycharmProjects\imageClassification\MyCNN'
#tf.compat.v1.summary.FileWriter
train_writer = tf.compat.v1.summary.create_file_writer(LOGDIR)
train_writer.add_graph(sess.graph)