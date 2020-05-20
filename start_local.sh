# NOTE: steps
# 1. check yaml file. is it correct? is distributed=True?
# 2. check CUDA_VISIBLE_DEVICES is consistent to nproc_per_node
# 3. change the lr according to num_gpu

# retrain and single GPU
# python main.py --gpu_id 0 --mode train --data ~/datasets/ilsvrc_12 --config config/mbv2/retrain.yaml \
#     --flops 43 --chcfg ./results/mbv2-211m/model_sample/expected_ch

# train ours cfg
# python main.py --gpu_id 0 --mode train --data ~/datasets/ilsvrc_12 --config config/mbv2/aps_retrain.yaml --flops 43 --chcfg ./APS_results/chansearch-enx2f9u-314M/ 

# retrain on multi-gpu on a singe node
python -u -m torch.distributed.launch --nproc_per_node=2 main.py \
  --port 6502 \
  --gpu_id 0,1 \
  --mode train \
  --data ~/datasets/ilsvrc_12 \
  --flops 43 \
  --config config/mbv2/retrain.yaml \
  --chcfg ./archs/expected_ch_300M/

#   --chcfg ./archs/chansearch-enx2f9u-314M/


# CUDA_VISIBLE_DEVICES=0 RANK=0 WORLD_SIZE=2 \
# python main.py \
#   --mode train \
#   --data ~/datasets/ilsvrc_12 \
#   --config config/mbv2/retrain.yaml \
#   --flops 43 \
#   --chcfg ./results/mbv2-300m/model_sample/expected_ch

# CUDA_VISIBLE_DEVICES=2 RANK=1 WORLD_SIZE=2 \
# python main.py \
#   --mode train \
#   --data ~/datasets/ilsvrc_12 \
#   --config config/mbv2/retrain.yaml \
#   --flops 43 \
#   --chcfg ./results/mbv2-300m/model_sample/expected_ch
