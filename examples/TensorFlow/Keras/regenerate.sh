#!/bin/bash
set -ex

python keras_model.py -nogui

python keras2MDFfunc_cal.py
