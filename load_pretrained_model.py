"""Example of loading a pre-trained APC model."""

import torch

from apc_model import APCModel
from utils import PrenetConfig, RNNConfig
# added by Sameer
import kaldiio
import sys


feats_scp = sys.argv[1]
segments = sys.argv[2]
scp_file = sys.argv[3]

ark_file = scp_file.replace('.scp', '.ark')
writer = kaldiio.WriteHelper('ark,scp:%s,%s' % (ark_file, scp_file))

if segments:
  reader = kaldiio.ReadHelper('scp:%s' % feats_scp, segments=segments)
else:
  reader = kaldiio.ReadHelper('scp:%s' % feats_scp)

def main():
  prenet_config = None
  rnn_config = RNNConfig(input_size=80, hidden_size=512, num_layers=3,
                         dropout=0., residual=True)  # Sameer Added residual=True
  pretrained_apc = APCModel(mel_dim=80, prenet_config=prenet_config,
                            rnn_config=rnn_config).cuda()

  pretrained_weights_path = '/data/sls/qcri/asr/sameer_v1/data/bs32-rhl3-rhs512-rd0-adam-res-ts1.pt'
  pretrained_apc.load_state_dict(torch.load(pretrained_weights_path))

  pretrained_apc.cuda()

  # Load data and perform your task ...
  i = 0
  for k, v in reader:
      i+=1
      v = torch.tensor(v).unsqueeze(0)[..., :80]  # b x t x f
      lengths = torch.tensor([v.shape[1]]).long()
      _, feats = pretrained_apc.forward(v, lengths)
      feats = feats[-1, :, :, :].detach().cpu().squeeze(0).numpy()
      writer(k, feats)
      if i % 100==0:
          print('Done %d' % i)

if __name__ == '__main__':
    main()
