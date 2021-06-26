#!/bin/bash
cd ~/dogifier

model_names=("dino_vits16" "dino_vits8" "dino_vitb16" "dino_vitb8")
batch_sizes=(512 512 256 256)
batch_sizes_finetune=(64 64 64 64)

for i in "${!model_names[@]}";
do
    echo ${model_names[$i]}

    echo ${batch_sizes[$i]}
    python main.py \
        model=dino \
        expr=${model_names[$i]} \
        datamodule.batch_size=${batch_sizes[$i]}

    echo ${batch_sizes[$i]}
    python main.py \
        model=dino \
        model.freeze_backbone=False \
        expr=${model_names[$i]}_finetune \
        datamodule.batch_size=${batch_sizes_finetune[$i]}
done
