## Tensorflow Object Detection API - Getting Started

### 0. Install CUDA and CUDNN

### 1. Install Anaconda 

https://www.anaconda.com/distribution/

### 2. Create virtual environment 

```
conda create -n tf_obj_api python=3.6 anaconda
```

Switch to newly created environment

```
conda activate tf_obj_api
```

### 3. Install Tensorflow

```
conda install tensorflow-gpu==1.9
```

### 4. Install Tensorflow's dependencies

```
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
pip install --user Cython
pip install --user contextlib2
pip install --user jupyter
pip install --user matplotlib
```

### 5. Download the tensorflow object detection API repository

If you don't have git installed - 

```
sudo apt install git
```

After installing git -

```
git clone https://github.com/tensorflow/models.git
cd models
git checkout r1.12.0
git pull
```

### 6. Protobuf compilation 

cd into the "models/research" directory and do the following -

```
protoc object_detection/protos/*.proto --python_out=.
```

If you get no output, that means that there is no error. You're good to go to the next step.

If you get an error, you might have an outdated version of protoc. Download the latest version online and use the file in the "bin" folder instead of the "protoc" in the above line.

### 7. Add libraries to Python Path

When running locally, the tensorflow/models/research/ and slim directories should be appended to PYTHONPATH. This can be done by running the following from tensorflow/models/research/ 

```
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

Note: This command needs to run from every new terminal you start. If you wish to avoid running this manually, you can add it as a new line to the end of your ~/.bashrc file, replacing `pwd` with the absolute path of tensorflow/models/research on your system.

### 8. Testing the installation 

You can test that you have correctly installed the Tensorflow Object Detection API by running the following command:

```
# From models/research/
python object_detection/builders/model_builder_test.py
```

Run the jupyter notebook "object_detection_tutorial" in the "models/research/object_detection" directory

To make the virtual environment available on the Jupyter Notebook - https://stackoverflow.com/questions/39604271/conda-environments-not-showing-up-in-jupyter-notebook

```
conda install nb_conda
conda install ipykernel
python -m ipykernel install --user --name tf_obj_api
```

## Training a custom object detector

### 1. Obtain/Create the object detection dataset with the bounding box labels. This should be in the VOC PASCAL format (images + xml annotation in VOC PASCAL for each image).

### 2. Convert the xml annotation to csv. 

```
trainDf = xml_to_df(trainPath)
testDf = xml_to_df(testPath)
trainDf.to_csv(trainCsvPath , index=None)
testDf.to_csv(testCsvPath, index=None)
```

### 3. Install object_detection module? (Wasn't required for me but might be required for others?)

```
# from the "models/research" directory
sudo python3 setup.py install
```

### 4. Generate TF Record

```
# once for training path and once for validation path
genTfr(outputPath, imageDir, csvInput, catDict) 
```

### 5. Create .pbtxt file

Create a file with an "item" entry (contains id and name) for every object in your object detection dataset.

```
item {
  id: 1
  name: 'Abyssinian'
}

item {
  id: 2
  name: 'american_bulldog'
}

item {
  id: 3
  name: 'american_pit_bull_terrier'
}

item {
  id: 4
  name: 'basset_hound'
}
```


Code to do this for you - 

```
catDict = {}
# tup[0] + 1 because category can't be 0 (its a placeholder is what I heard from sentdex's video). Always start from 1. 
list(map(lambda tup : catDict.update({tup[1]:tup[0] + 1}) , enumerate( trainDf['class'].unique() ) ) )
genTfr(trainTfrPath, trainPath, trainCsvPath, catDict)
genTfr(testTfrPath, testPath, testCsvPath, catDict)

writePbtext(catDict, pbTextPath)
```
MAKE SURE YOU USE THE SAME "catDict" (same mapping) FOR BOTH "genTfr" AND "writePbtext".

### 6. Create/Obtain configuration file + pretrained model

Link to configuration files - https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs

Link to model zoo - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

Link to pbtxt - https://github.com/tensorflow/models/tree/master/research/object_detection/data

Replace the essential stuff in config file with your own configuration. Possible "essential stuff" -

```
num_classes: $NUM_CLASSES

train_config: {
  batch_size: $BATCH_SIZE

fine_tune_checkpoint: "$FINETUNED_MODEL_PATH"

train_input_reader: {
  tf_record_input_reader {
    input_path: "$TRAIN_TFR_PATH"
  }
  label_map_path: "$PBTEXT_PATH"
}

eval_config: {
  metrics_set: "coco_detection_metrics"
  num_examples: $NUM_TEST_SAMPLES
}

eval_input_reader: {
  tf_record_input_reader {
    input_path: "$TEST_TFR_PATH"
  }
  label_map_path: "$PBTEXT_PATH"
  shuffle: false
  num_readers: 1
}

```

### 7. Use the training command to start the training

```
import subprocess

trainingCmd = 'python {2} --logtostderr --train_dir={0} --pipeline_config_path={1}'.format(
        os.path.abspath(tfTrainOutDir), 
        os.path.abspath(configOutPath), 
        os.path.join(tfObjectDetFolder, 'legacy', 'train.py'))
process = subprocess.Popen(trainingCmd, shell=True, stdout=subprocess.PIPE)
        process.wait()
```

The command on the terminal would look something like - 

```
python path_to/train.py --logtostderr --train_dir=path_to_train_dir --pipeline_config_path=path_to_config_file
```
"path_to_config_file" is the ".config" file obtained in step 6. 
"path_to_train_dir" is the directory where the output of the training is stored.

How does the model know where the training and validation images and bounding boxes are? These will be inside the train.tfrecord and valid.tfrecord. 
How does the model know where these tfrecord files are? The path to these will be inside the config file (along with a lot of other configurations which are basically inputs provided by the user that the model would expect)


### 8. Export inference graph

```
inferencePath = os.path.join(destn, inferenceDir)
inferenceCmd = 'python {3} --input_type image_tensor --pipeline_config_path {0} --trained_checkpoint_prefix {1} --output_directory {2}'.format(
    os.path.abspath(configOutPath), 
    os.path.abspath(tfTrainOutDir) + '/' + MODEL_FILE_PLACEHOLDER,
    os.path.abspath(inferencePath),
    os.path.join(tfObjectDetFolder, 'export_inference_graph.py'))

#once training is complete - 
inferenceCmd = inferenceCmd.replace(MODEL_FILE_PLACEHOLDER, _findLatestModel(tfTrainOutDir, modelFilePrefix))
process = subprocess.Popen(inferenceCmd, shell=True, stdout=subprocess.PIPE)
process.wait()
```

```
python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/ssd_mobilenet_v1_pets.config \
    --trained_checkpoint_prefix training/model.ckpt-10856 \
    --output_directory mac_n_cheese_inference_graph
```




### Errors and things to note

When using models with a lot of classes, do not use the code from the sample provided in the object detection sample ipynb. This will result in incorrect indices.

Source - https://github.com/tensorflow/models/issues/6371

"The cause of the problem is the conversion of the category indices to uint8 in the following line:
output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)

While this is ok for 80-class COCO, it causes overflow in datasets with a large number of classes, such as OID. If you change it to np.int64, it works correctly."
