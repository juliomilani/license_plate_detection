Pré-requisitos para testar:
 * Python 3
 * Tensorflow
 * Numpy
 * OpenCV

Imagens teste se encontram na pasta teste_imgs e a resposta do sistema em teste_imgs_out

Como detectar placas em outras imagens:
 
git clone https://github.com/juliomilani/license_plate_detection.git
cd license_plate_detection

#instala todos os pacotes necessários
pip install requirements.txt

python plate_finder_tf.py --path_in teste_imgs --path_out teste_imgs_out

--path_in: Pasta contendo arquivos de imagem para achar placas
--path_out: Pasta onde será salvo as imagens com as placas encontradas



Para treinar com outros dados basta seguir o tutorial:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md

Ao seguir o tutorial 

How to test a new model in tensorflow object detection API:
1) Configuring the object_detection/protos/pipeline.proto
		ex: ssd_mobilenet_v2_placas/data/pipeline.config
2) Programming create_tf_record.py

tensorboard --logdir=PATH_TO_LICENSE_PLATE_DETECTION/models/1306

python model_main.py --pipeline_config_path=PATH_TO_LICENSE_PLATE_DETECTION/models/1306/pipeline.config --model_dir=PATH_TO_LICENSE_PLATE_DETECTION/models/1306 --num_train_steps=50000 --sample_1_of_n_eval_examples=1 --alsologtostderr

python C:\tensorflow\plate_detector2\eval.py --logtostderr --checkpoint_dir=PATH_TO_LICENSE_PLATE_DETECTION/models/1306/train --eval_dir=PATH_TO_LICENSE_PLATE_DETECTION/models/1306/eval --pipeline_config_path=PATH_TO_LICENSE_PLATE_DETECTION/models/1306/pipeline.config
python C:\tensorflow\plate_detector2\train.py --logtostderr --train_dir=C:\tensorflow\plate_detector2\models\1306\train --pipeline_config_path=C:\tensorflow\plate_detector2\models\1306\pipeline.config

python C:\tensorflow\plate_detector2\export_inference_graph.py --input_type=image_tensor --pipeline_config_path=PATH_TO_LICENSE_PLATE_DETECTION/models/1306/pipeline.config --trained_checkpoint_prefix=C:\tensorflow\plate_detector2\models\1306\train\model.ckpt-7252 --output_directory=PATH_TO_LICENSE_PLATE_DETECTION\models\1306\out-7252

Important fixes:

schedule {
step: 0
learning_rate: 0.000199999994948
}

removed from pipeline.config
