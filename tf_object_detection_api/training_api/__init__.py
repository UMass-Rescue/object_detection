import os
import shutil
import numpy as np

import glob
import pandas as pd
import xml.etree.ElementTree as ET
from functools import reduce
from generate_tfrecord import genTfr
import subprocess