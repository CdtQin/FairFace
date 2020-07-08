#!/bin/sh

GPU_NUM=8
GPU_DEVICE='0,1,2,3,4,5,6,7'
BATCH_SIZE=2048

MODEL_PATH=graph.proto

KEY=key.fea

DATA_PATHS=(
data.list
)

OUTPUT_DIR=/OUTPUT_DIR
OUTPUT_PATHS=(
${OUTPUT_DIR}/test_${KEY}
)

SUFFIX='_new'
extract_img_feature()
{
    i=$1
    DATA_PATH=${DATA_PATHS[$i]}
    OUTPUT_PATH=${OUTPUT_PATHS[$i]}
    echo "start data: ${DATA_PATH} >> ${OUTPUT_PATH}"
    nohup python extract_fea_flip.py --batch_size=${BATCH_SIZE} --num_gpus=${GPU_NUM} \
                       --data_path=${DATA_PATH} --pb_path=${MODEL_PATH} \
                       --output_path=${OUTPUT_PATH} \
                       &>logs/extract_feature_tmp.log &
}

echo "START"
for ((i=0;i<${#DATA_PATHS[*]};i++))
do
    CUDA_VISIBLE_DEVICES=${GPU_DEVICE} extract_img_feature $i
    PID=$!
    wait $PID
done
