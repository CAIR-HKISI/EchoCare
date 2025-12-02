now=$(date +"%Y%m%d_%H%M%S")
logdir=runs/logs_echocare
mkdir -p $logdir

torchrun --nproc_per_node 8 --master_port=28802 echocore_train.py \
         --logdir  $logdir \
         --datadir /scratch/esg8sdce/esg8sdeuser02/pretrain_datasets

