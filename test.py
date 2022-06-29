import pandas as pd
import torch
from torch.utils.data import Dataset
from model import PQRNN
from dataset import create_dataloaders,DummyDataset
model = PQRNN(b=128,d=64,output_size=6,dropout=0.5,lr=1e-3,num_layers=1,fc_sizes=[128,64],rnn_type='GRU',multilabel=True)#.load_from_checkpoint(checkpoint_path='checkpoints.ckpt',map_location=torch.device('cpu'),hparams_file='lightning_logs/default/version_0/hparams.yaml')

df = pd.read_csv('data/test.csv')
#test_dataset = DummyDataset(df[['comment_text']].to_dict('records'),has_label=False,feature_size=128*2,add_eos_tag=True,add_bos_tag=True,max_seq_len=512,label2index=None)

dataloader = create_dataloaders(task='toxic',label2index=None,batch_size=32,feature_size=128*2,data_path='data')
#dataloader = create_dataloaders(test_dataset)
for test in dataloader:
  for proj in test:
    #projections,lengths = batch
    print(proj[0],proj[1],proj[2])
    break