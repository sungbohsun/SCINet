# SCINet

[![Arxiv link](https://img.shields.io/badge/arXiv-Time%20Series%20is%20a%20Special%20Sequence%3A%20Forecasting%20with%20Sample%20Convolution%20and%20Interaction-%23B31B1B)](https://arxiv.org/pdf/2106.09305.pdf)

![pytorch](https://img.shields.io/badge/-PyTorch-%23EE4C2C?logo=PyTorch&labelColor=lightgrey)

## 環境建置
```
git clone https://github.com/sungbohsun/SCINet
cd SCINet
pip install -r .\requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

## 啟動streamlit
```
streamlit run .\run.py
```

## streamlit 訓練與驗證工具
![pic1](https://github.com/sungbohsun/SCINet/blob/main/demo/demo1.png)
![pic2](https://github.com/sungbohsun/SCINet/blob/main/demo/demo2.png)
![pic3](https://github.com/sungbohsun/SCINet/blob/main/demo/demo3.png)
![pic4](https://github.com/sungbohsun/SCINet/blob/main/demo/demo4.png)
![pic5](https://github.com/sungbohsun/SCINet/blob/main/demo/demo5.png)
 2020-07-20

## tensorboard訓練紀錄
```
tensorboard --logdir=./event
```
##### Parameter highlights

| Parameter Name | Description                  | Parameter in paper | Default                    |
| -------------- | ---------------------------- | ------------------ | -------------------------- |
| root_path      | The root path of subdatasets | N/A                | './datasets/ETT-data/ETT/' |
| data           | Subdataset                   | N/A                | ETTh1                      |
| pred_len       | Horizon                      | Horizon            | 192                         |
| seq_len        | Look-back window             | Look-back window   | 96                         |
| batch_size     | Batch size                   | batch size         | 16                         |
| lr             | Learning rate                | learning rate      | 1e-3                     |
| hidden-size    | hidden expansion             | h                  | 1                          |
| levels         | SCINet block levels          | L                  | 3                          |
| stacks         | The number of SCINet blocks  | K                  | 1                          |
