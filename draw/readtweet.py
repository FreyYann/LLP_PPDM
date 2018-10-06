from sklearn import metrics
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
module_path = os.path.abspath(os.path.join('/home/virgile/adaptive_confound/'))
if module_path not in sys.path:
    sys.path.append(module_path)
import adaptive_confound.utils as acu

datapath = "/data/virgile/confound/adaptive/in/"
tw_ylzg = acu.read_pickle(os.path.join(datapath, "twitter_dataset_y=location_z=gender.pkl"))
tw_ygzl = acu.read_pickle(os.path.join(datapath, "twitter_dataset_y=gender_z=location.pkl"))

