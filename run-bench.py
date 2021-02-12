#!/usr/bin/env python3

import sys
import os
import subprocess


models = [
    # 'VGG11',
    # 'VGG13',
    # 'VGG16',
    # 'VGG19',
    'MobileNet',
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
<<<<<<< HEAD
            os.system("python main.py --model {} --groups {} --overlap {}".format(model, grp, oap))
            # p = subprocess.Popen("python main.py --model {} --groups {} --overlap {}".format(model, grp, oap), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
            # print("time: {}".format(p.split("\n")[-2].split(':')[-1]))
=======
            os.system("python main.py --model {} --groups {} --overlap {}".format(model, grp, oap))
>>>>>>> 75eee9d4f0e266e561f7833555d9997ffc29a9c6
