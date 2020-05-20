# NOTE: steps
# 0. change parameter in the right yaml file (bs, lr)
# 1. check yaml file. is it correct? is distributed=True?
# 2. check CUDA_VISIBLE_DEVICES is consistent to nproc_per_node
# 3. change the lr according to num_gpu

## =========================== mobilenet_v2 train from scratch  =========================
sh seven.sh \
  -n=1 \
  --gpu_id 0 \
  --port 6512 \
  --mode train \
  --data ~/datasets/ilsvrc_12 \
  --flops 43 \
  --config config/mbv2/retrain.yaml \
  --chcfg ./archs/expected_ch_300M/

#   --config config/mbv2/aps_retrain.yaml \
#   --chcfg ./archs/chansearch-enx2f9u-314M

