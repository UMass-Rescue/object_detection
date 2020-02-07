import os
import shutil
import numpy as np
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from functools import reduce
from .generate_tfrecord import genTfr
import subprocess
from .util import vocTrainTestSplit, xml_to_df

MODEL_FILE_PLACEHOLDER = 'YOUR_MODEL_FILE'

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

def get_file_paths(destn, trainFolName, testFolName, trainCsvName, testCsvName, trainTfrName, testTfrName, trainOutDirName, pbTextName):
    trainPath, testPath = os.path.join(destn, trainFolName), os.path.join(destn, testFolName)
    trainCsvPath, testCsvPath = os.path.join(destn, trainCsvName), os.path.join(destn, testCsvName)
    trainTfrPath, testTfrPath = os.path.join(destn, trainTfrName), os.path.join(destn, testTfrName)
    tfTrainOutDir = os.path.join(destn, trainOutDirName)
    pbTextPath = os.path.join(destn, pbTextName)
    return trainPath, testPath, trainCsvPath, testCsvPath, trainTfrPath, testTfrPath, tfTrainOutDir, pbTextPath

def train(imgDir:str, preTrainedModelPath:str, tfObjectDetFolder:str, destn:str = None, ratio:float=0.8, imgFmt:str='.jpg', testFolName='valid', trainFolName='train', trainCsvName='train.csv', testCsvName='valid.csv', trainTfrName='train.record', testTfrName='valid.record', trainOutDirName='trainOutput', pbTextName='obj_det.pbtxt', configInPath='ssd_mobilenet_v1_pets.config', configOutName='ssd_mobilenet_v1.config', inferenceDir='inference_graph', batchSize=24, modelFilePrefix='model.ckpt'):
    '''
    imgDir :: string - directory containing images and labels labelled in PASCAL VOC format
    preTrainedModelPath - string containing the path to the pretrained model that needs to be fine tuned
    tfObjectDetFolder - string containing the path to the "models/research/object_detection" directory in the TF Object Detection API.
    destn - the directory where the outputs generated by this function will be placed. If None, will create directory "../TFRConv" and make that the output directory
    ratio - ratio of the training and the validation split 
    imgFmt - string containing the format of the images in the "imgDir"
    '''
    destn = os.path.join(imgDir, '../TFRConv/') if destn is None else destn
    vocTrainTestSplit(imgDir, destn, ratio=ratio, createDir=True, imgFmt=imgFmt, testFolName=testFolName, trainFolName=trainFolName)

    trainPath, testPath, trainCsvPath, testCsvPath, trainTfrPath, testTfrPath, tfTrainOutDir, pbTextPath = get_file_paths(destn, trainFolName, 
            testFolName, trainCsvName, testCsvName, trainTfrName, testTfrName, trainOutDirName, pbTextName)

    configInPath, configOutPath = configInPath, os.path.join(destn, configOutName)
    trainDf = xml_to_df(trainPath)
    testDf = xml_to_df(testPath)
    trainDf.to_csv(trainCsvPath , index=None)
    testDf.to_csv(testCsvPath, index=None)
    # TODO - the following will work well for for just one category. But if there are multiple categories
    # what if there are some categories which got sent to the test set without a single instance of the same class 
    # in the training set?
    catDict = {}
    # tup[0] + 1 because category can't be 0 (its a placeholder is what I heard from sentdex's video). Always start from 1. 
    list(map(lambda tup : catDict.update({tup[1]:tup[0] + 1}) , enumerate( trainDf['class'].unique() ) ) )
    genTfr(trainTfrPath, trainPath, trainCsvPath, catDict)
    genTfr(testTfrPath, testPath, testCsvPath, catDict)

    # TODO - can use exist_ok=True
    if not os.path.exists(tfTrainOutDir):
        os.makedirs(tfTrainOutDir)

    writePbtext(catDict, pbTextPath)
    # TODO - AGAIN, assumes that trainDf contains all the classes. Might not be the case, especially in small datasets. 
    # i.e. there might be a class in validation set which wasn't present in the training set.
    numClasses, numTestSamples = len(trainDf['class'].unique()), testDf.shape[0]
    configInPath = os.path.join(__location__, configInPath)
    writeConfigFile(configInPath, configOutPath, 
            genConfPlaceholders(numClasses, 
                os.path.abspath(preTrainedModelPath), 
                os.path.abspath(trainTfrPath), 
                os.path.abspath(testTfrPath), 
                os.path.abspath(pbTextPath), 
                numTestSamples,
                batchSize=batchSize)
            )
    trainingCmd = 'python {2} --logtostderr --train_dir={0} --pipeline_config_path={1}'.format(
        os.path.abspath(tfTrainOutDir), 
        os.path.abspath(configOutPath), 
        os.path.join(tfObjectDetFolder, 'legacy', 'train.py'))
    inferencePath = os.path.join(destn, inferenceDir)
    inferenceCmd = 'python {3} --input_type image_tensor --pipeline_config_path {0} --trained_checkpoint_prefix {1} --output_directory {2}'.format(
        os.path.abspath(configOutPath), 
        os.path.abspath(tfTrainOutDir) + '/' + MODEL_FILE_PLACEHOLDER,
        os.path.abspath(inferencePath),
        os.path.join(tfObjectDetFolder, 'export_inference_graph.py'))

    print('For training (from the object_detection/legacy directory) :')
    print(trainingCmd)
    print()
    print('To generate inference graph (after training) (from the object_detection directory) :')
    print(inferenceCmd)

    #python train.py --logtostderr --train_dir=/home/prasannals/Downloads/handsup/data --pipeline_config_path=/home/prasannals/Downloads/handsup/data/TFRConv/ssd_mobilenet_v1_pets.config
    #python export_inference_graph.py --input_type image_tensor --pipeline_config_path /home/prasannals/Downloads/handsup/data/TFRConv/ssd_mobilenet_v1_pets.config --trained_checkpoint_prefix /home/prasannals/Downloads/handsup/data/model.ckpt-4733 --output_directory /home/prasannals/Downloads/handsup/data/TFRConv/inference_graph

    try:
        process = subprocess.Popen(trainingCmd, shell=True, stdout=subprocess.PIPE)
        process.wait()
    except KeyboardInterrupt:
        print('Training done')
        inferenceCmd = inferenceCmd.replace(MODEL_FILE_PLACEHOLDER, findLatestModel(tfTrainOutDir, modelFilePrefix))
        process = subprocess.Popen(inferenceCmd, shell=True, stdout=subprocess.PIPE)
        process.wait()

        print('Inference graph written to {0}'.format(inferencePath))

        print('Training command used - ')
        print(trainingCmd)
        print('Inference command used - ')
        print(inferenceCmd)

        return (os.path.abspath(inferencePath), os.path.abspath(pbTextPath), numClasses)


def findLatestModel(tfTrainOutDir, modelFilePrefix):
    return modelFilePrefix + '-' + str(sorted(list( 
        map( lambda f: int(f[:f.index('.')]) , 
            map( lambda f: f[f.index('-')+1:] ,
                filter(lambda f: f.startswith(modelFilePrefix), os.listdir(tfTrainOutDir))))))[-1])

def writeConfigFile(inPath, outPath, values, prefix='$'):
    confStr = None
    with open(inPath) as f:
        confStr = f.read()
    with open(outPath, 'w') as f:
        out = configFileFormat(confStr, values, prefix=prefix)
        f.write(out)

def configFileFormat(inpStr, values, prefix="$"):
    return reduce(lambda s, key: s.replace(prefix + key, str(values[key]) ), values.keys(), inpStr)

def genConfPlaceholders(numClasses, preTrainedModelPath, trainTfrPath, testTfrPath, pbTextPath, numTestSamples, batchSize=24 ):
    return {
        'NUM_CLASSES':numClasses,
        'FINETUNED_MODEL_PATH':preTrainedModelPath,
        'TRAIN_TFR_PATH':trainTfrPath,
        'TEST_TFR_PATH':testTfrPath,
        'PBTEXT_PATH':pbTextPath,
        'NUM_TEST_SAMPLES':numTestSamples,
        'BATCH_SIZE':batchSize
    }

def writePbtext(catDict, path):
    '''
    catDict :: dictionary<string, int> - dictionary mapping category name to a unique integer
    path :: string - output path for the pbtext
    '''
    with open(path, 'w') as f:
        f.write(dictToPbtextStr(catDict))

def dictToPbtextStr(catDict):
    '''
    catDict :: dictionary<string, int> - dictionary mapping category name to a unique integer

    return :: string - the string representation of the pbtext file for the given catDict
    '''
    return reduce(lambda s1, s2: s1 + '\n' + s2, map( lambda it: catToPbtextStr(it[0], it[1]) , catDict.items()))

def catToPbtextStr(name, id):
    '''
    name :: string - category name
    id :: string - id name

    return :: string - string representation of the entry in pbtext for the passed in name and id
    '''
    return 'item {{\n  name: "{0}"\n  id: {1}\n}}'.format(name, id)