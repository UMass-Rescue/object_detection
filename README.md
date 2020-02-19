# object_detection

Moved Object Detection API to a seperate pip installable library - https://github.com/prasannals/tf_object_detection_util

Moved Object Summary to a seperate pip installable library - https://github.com/UMass-Rescue/object_summary

Getting started - 

1. Install required libraries

```
pip install opencv-python
pip install tf_object_detection_util
pip install object_summary
```

2. Clone this repo

```
git clone https://github.com/UMass-Rescue/object_detection.git
```

3. (If required) add tensorflow object detection api to your path. In case you're not able to set the PYTHONPATH environment variable for some reason, you can directly add the following lines to the start of the notebook where you'll be using any of the above libraries in order to make Tensorflow Object Detection library available for the code

```
import sys
sys.path.append('/home/jupyter/models/research/') # replace this path with your systems path to the "models/research" directory
sys.path.append('/home/jupyter/models/research/slim/') # replace this path with your systems path to "models/research/slim" directory
```

4. Download faster rcnn from http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12.tar.gz and extract the contents.

5. Download ".pbtxt" file for our faster rcnn model from https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/oid_v4_label_map.pbtxt 

6. Replace "PATH_TO_FROZEN_GRAPH" and "PATH_TO_LABELS" variables in "load_faster_rcnn_open_images" method to point to your own files. "PATH_TO_FROZEN_GRAPH" should point to the "frozen_inference_graph.pb" file obtained from step 4. "PATH_TO_LABELS" should point to the "oid_v4_label_map.pbtxt" downloaded in step 5.

7. Run the rest of the cells to obtain the results on the sample data provided.

### Datasets used - 

* <strong>Object detection datasets</strong>
  * Open Images - https://opensource.google/projects/open-images-dataset

* <strong>Classification Datasets</strong>
  * Cola Bottle Identification - https://www.kaggle.com/deadskull7/cola-bottle-identification
  * Messy vs Clean room - https://www.kaggle.com/cdawn1/messy-vs-clean-room

* Images from unsample.net
