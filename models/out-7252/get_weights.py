import tensorflow as tf
from tensorflow.python.platform import gfile

GRAPH_PB_PATH = r'C:\tensorflow\plate_detector2\models\1306\out-7252\frozen_inference_graph.pb' #path to your .pb file
with tf.Session() as sess:
  print("load graph")
  with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    graph_nodes=[n for n in graph_def.node]

wts = [n for n in graph_nodes if n.op=='Const']

from tensorflow.python.framework import tensor_util


txt_file = open("weights.txt","w") 

for n in wts:
    print("Name of the node - %s" % n.name)
    print( "Value - ") 
    print(tensor_util.MakeNdarray(n.attr['value'].tensor))
    txt_file.write("Name of the node - %s" % n.name)
    txt_file.write('\n')
    txt_file.write( "Value - ") 
    txt_file.write('\n')
    txt_file.write(str(tensor_util.MakeNdarray(n.attr['value'].tensor)))
    txt_file.write('\n')

txt_file.close() 