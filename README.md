<<<<<<< HEAD
Pré-requisitos para testar:
 * Python 3
 * Tensorflow
 * Numpy
 * OpenCV

 Alterar pasta in_folder e out_folder em plate_finder_tf.py e executar o arquivo para localizar as placas de todas as imagens na pasta in_folder




=======
Resultados se encontram na pasta imgs_out


-------------------------
>>>>>>> 8ef711af0cbb860e5bb3310def09b1b22d49b9c5
How to test a new model in tensorflow object detection API:
1) Configuring the object_detection/protos/pipeline.proto
		ex: ssd_mobilenet_v2_placas/data/pipeline.config
2) Programming create_tf_record.py




tensorboard --logdir=C:/tensorflow/plate_detector2/models/1306

python model_main.py --pipeline_config_path=C:/tensorflow/plate_detector2/models/1306/pipeline.config --model_dir=C:/tensorflow/plate_detector2/models/1306 --num_train_steps=50000 --sample_1_of_n_eval_examples=1 --alsologtostderr

python C:\tensorflow\plate_detector2\eval.py --logtostderr --checkpoint_dir=C:/tensorflow/plate_detector2/models/1306/train --eval_dir=C:/tensorflow/plate_detector2/models/1306/eval --pipeline_config_path=C:/tensorflow/plate_detector2/models/1306/pipeline.config
python C:\tensorflow\plate_detector2\train.py --logtostderr --train_dir=C:\tensorflow\plate_detector2\models\1306\train --pipeline_config_path=C:\tensorflow\plate_detector2\models\1306\pipeline.config


python C:\tensorflow\plate_detector2\export_inference_graph.py --input_type=image_tensor --pipeline_config_path=C:/tensorflow/plate_detector2/models/1306/pipeline.config --trained_checkpoint_prefix=C:\tensorflow\plate_detector2\models\1306\train\model.ckpt-7252 --output_directory=C:\tensorflow\plate_detector2\models\1306\out-7252





fixes:

schedule {
step: 0
learning_rate: 0.000199999994948
}

removed from pipeline.config

File "C:\tensorflow\models\research\object_detection\utils\object_detection_evaluation.py", line 213, in _build_metric_names
category_name = unicode(category_name, 'utf-8')
NameError: name 'unicode' is not defined
"unicode() --> str()
