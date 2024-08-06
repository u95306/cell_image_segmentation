#!/bin/bash

python copy_image.py

cp -r Dataset/training_set/positive/. Dataset/test_set/positive/
cp -r Dataset/training_set/negative/. Dataset/test_set/negative/