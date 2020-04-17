# ALPR: Automatic License Plate Reader

Yet another CNN's based Automatic Plate Detection.

![Plate Image](https://github.com/juliomilani/license_plate_detection/blob/master/teste_imgs_out/prefeitura_rio.png "Input image")

Vídeo rede funcionando: https://youtu.be/7zgupFMNE1w

A rede foi treinada usando o data-set UTFPR-ALPR. Disponível, sob demanda:
https://web.inf.ufpr.br/vri/databases/ufpr-alpr/

A arquitetura da rede é: 
faster_rcnn_inception_v2_coco_2018_01_28

## Getting Started

To test with your own images you can clone this repo and run stremlit:

```
$ git clone https://github.com/juliomilani/license_plate_detection.git
$ cd license_plate_detection

$ #instala todos os pacotes necessários
$ pip install requirements.txt
$ streamlit run st_app_ocr.py
```
Open the link in a browser.


## Running on multiple images

```
python plate_finder_tf.py --path_in IN_FOLDER --path_out teste_imgs_out
--path_in: In folder, containing only images
--path_out: Folders where the annotated images will be stored

```

## Retraining with your own dataset:

Follow the tutorial this on creating a tf_record (you can use create_tf_record.py as a model):
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md

Then running these commands:
```
tensorboard --logdir=PATH_TO_LICENSE_PLATE_DETECTION/models/1306

python model_main.py --pipeline_config_path=PATH_TO_LICENSE_PLATE_DETECTION/models/1306/pipeline.config --model_dir=PATH_TO_LICENSE_PLATE_DETECTION/models/1306 --num_train_steps=50000 --sample_1_of_n_eval_examples=1 --alsologtostderr

python eval.py --logtostderr --checkpoint_dir=PATH_TO_LICENSE_PLATE_DETECTION/models/1306/train --eval_dir=PATH_TO_LICENSE_PLATE_DETECTION/models/1306/eval --pipeline_config_path=PATH_TO_LICENSE_PLATE_DETECTION/models/1306/pipeline.config
python train.py --logtostderr --train_dir=C:\tensorflow\plate_detector2\models\1306\train --pipeline_config_path=C:\tensorflow\plate_detector2\models\1306\pipeline.config

python export_inference_graph.py --input_type=image_tensor --pipeline_config_path=PATH_TO_LICENSE_PLATE_DETECTION/models/1306/pipeline.config --trained_checkpoint_prefix=C:\tensorflow\plate_detector2\models\1306\train\model.ckpt-7252 --output_directory=PATH_TO_LICENSE_PLATE_DETECTION\models\1306\out-7252
```

When following the tutorial there's one change you have to make:

Remove this part from pipeline.config: (I learned it the hardway)
```
schedule {
step: 0
learning_rate: 0.000199999994948
}
```





## Authors

* **Julio Milani de Lucena** LAPSI - UFRGS - julio.lucena@ufrgs.br
Orientador: Altamiro Susin

