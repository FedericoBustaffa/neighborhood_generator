#!/bin/bash

source .env/bin/activate

models=("RandomForestClassifier" "SVC" "MLPClassifier")

for model in ${models[@]}; do
    if [ $model = "RandomForestClassifier" ]; then
        workers=16
    else
        workers=8
    fi
    python complete.py $model $workers ${model}_expl &
done
