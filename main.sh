#!/bin/bash

# configure pip to use internal source
unset http_proxy
unset https_proxy
mkdir -p ~/.pip/ \
    && echo "[global]"                                              > ~/.pip/pip.conf \
    && echo "index-url = http://mirror-sng.oa.com/pypi/web/simple/" >> ~/.pip/pip.conf \
    && echo "trusted-host = mirror-sng.oa.com"                      >> ~/.pip/pip.conf
cat ~/.pip/pip.conf

# install Python packages with Internet access
pip install torchvision
pip install easydict
pip install pyyaml

# add the current directory to PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:`pwd`
export LD_LIBRARY_PATH=/opt/ml/disk/local/cuda/lib64:$LD_LIBRARY_PATH

LOG_DIR=/opt/ml/disk/dmcp_results
mkdir -p ${LOG_DIR}

# execute the main script
nvidia-smi # take a look at what GPUs

# mkdir models
EXTRA_ARGS=`cat ./extra_args`
N_GPU=`cat ./ngpu`

# copy ilsvrc dataset to /opt/ml/env
tmp_str=`cat ./extra_args | grep ilsvrc`
if [[ "${tmp_str}" != "" ]]
then
  # copy imagenet data to local machine
  echo "Running Imagenet experiments. First copy data to /opt/ml/env/"
  mkdir -v /opt/ml/env/datasets
  cp /opt/ml/disk/datasets/imagenet.tar.gz /opt/ml/env/datasets/
  echo "Copy done. Unzipping..."
  tar -xf /opt/ml/env/datasets/imagenet.tar.gz -C /opt/ml/env/datasets/
  echo "Unzipping done."
  python -u -m torch.distributed.launch --nproc_per_node=${N_GPU} main.py --save_path ${LOG_DIR} ${EXTRA_ARGS}
else
  python main.py --save_path ${LOG_DIR} ${EXTRA_ARGS}
fi


# remove *.pyc files
find . -name "*.pyc" -exec rm -f {} \;
