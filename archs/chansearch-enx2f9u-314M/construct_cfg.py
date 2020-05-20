import numpy as np

cfg = [40,40,20,60,60,30,72,72,30,90,90,40,120,120,40,120,120,40,96,96,80,192,192,80,144,144,80,240,240,80,480,480,120,432,432,120,576,576,120,432,432,200,1200,1200,200,960,960,200,1200,1200,400,1600]
chcfg = {}
chcfg['conv1'] = cfg[0]
chcfg['conv_last'] = tuple(cfg[-2:])

layer = 0
num_iter = 0
while True:
  tmp_cfg = {}
  if num_iter == 0:
    for idx in range(1,3):
      tmp_cfg['conv%s'%(str(idx+1))] = tuple(cfg[layer:layer+2])
      layer += 1
  else:
    for idx in range(3):
      tmp_cfg['conv%s'%(str(idx+1))] = tuple(cfg[layer:layer+2])
      layer += 1

  chcfg[str(num_iter)] = tmp_cfg
  if num_iter == 16:
    break
  else:
    num_iter += 1

assert len(list(chcfg.values())) == 19
print(layer)
print(chcfg)
np.save('sample.npy', chcfg, allow_pickle=True)
