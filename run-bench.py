#!/usr/bin/env python3

import sys
import os
import subprocess


models = [
    # 'VGG11',
    # 'VGG13',
    'VGG16',
    # 'VGG19',
    # 'MobileNet',
    # 'ResNet18',
    # 'ResNet34',
    # 'ResNet50',
]

groups = [
    # 1,
    2,
    # 4,
    # 8
]

overlaps = [
    # 0.25,
    # 0.33,
    0.50,
    # 0.75
]
for model in models:
    for grp in groups:
        for oap in overlaps:
            print("=> {}, g: {}, o: {}".format(model, grp, oap))
            os.system("python main.py --model {} --groups {} --overlap {}".format(model, grp, oap))