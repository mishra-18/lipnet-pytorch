## lipnet-pytorch
The code is based on the paper LipNet: End-to-End Sentence-level Lipreading. LipNet utilizes 3D Convolutions and Recurrent Units for make sentence level prediction from through extracting lip movment features from the input frames.
This implementation provides 3DConv-Bi-LSTM over the 3DConv-GRU model along with few other with other a few other model with varying complexity. CTC loss is used to deal with variable length of input alignments (spoken sentences). The model weights are initialised with the same (he) initialization as proposed in the paper.



<p align="center">
  
  <img src="https://github.com/mishra-18/lipnet-pytorch/assets/155224614/f6da2320-6e78-475f-9885-bd405d6c10d9" alt="Image Description" width="600"/>
</p>


## Training

venv suggested..

```
python -m venv lipenv
source lipenv/bin/activate
```
Install gdown for downloading the [dataset](https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL) from drive
```
pip install gdown
```

#### Train your model

```
python main.py --epoch 300 \
               --lr 0.001  \
               --hidden_size 256  \
               --model lipnet-lstm \
               --batch 16 \
               --workers 4
```

#### Reference
```
@article{assael2016lipnet,
  title={LipNet: End-to-End Sentence-level Lipreading},
  author={Assael, Yannis M. and Shillingford, Brendan and Whiteson, Shimon and de Freitas, Nando},
  journal={arXiv preprint arXiv:1611.01599},
  year={2016}
}
```
