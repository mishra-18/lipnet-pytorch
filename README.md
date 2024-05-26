## lipnet-pytorch
The code is divided into two parts. Loading and processing the video frames and training the model with CTC Loss.

#### Reference
```
@article{assael2016lipnet,
  title={LipNet: End-to-End Sentence-level Lipreading},
  author={Assael, Yannis M. and Shillingford, Brendan and Whiteson, Shimon and de Freitas, Nando},
  journal={arXiv preprint arXiv:1611.01599},
  year={2016}
}
```

<p align="center">
  <img src="https://github.com/mishra-18/lipnet-pytorch/assets/155224614/f6da2320-6e78-475f-9885-bd405d6c10d9" alt="Image Description" width="600"/>
</p>


## Training

venv suggested..

```
python -m venv lipenv
source lipenv/bin/activate
```
Clone the repo and navigate..
```
pip install requirements.txt
```

#### Train your model

```
python main.py --epoch 300 \
               --lr 0.001  \
               --hidden_size 256  \
               --model {default)lipnet-lstm \
               --batch 16 \
               --workers 4
```
