# SCINet

[![Arxiv link](https://img.shields.io/badge/arXiv-Time%20Series%20is%20a%20Special%20Sequence%3A%20Forecasting%20with%20Sample%20Convolution%20and%20Interaction-%23B31B1B)](https://arxiv.org/pdf/2106.09305.pdf)

![pytorch](https://img.shields.io/badge/-PyTorch-%23EE4C2C?logo=PyTorch&labelColor=lightgrey)


## demo.ipynb 範例、資料前處理與視覺化工具
![alt text](https://github.com/sungbohsun/SCINet/blob/main/demo.png)
 訓練使用csv 路徑為.\datasets\ETT-data\TrainOneDay.csv.

## 丟棄NA與std=0欄位
```
data = pd.read_csv(r'datasets\ETT-data\TrainOneDay.csv')
data = data.dropna(axis=0).drop(data.std()[(data.std() == 0)].index, axis=1)
```

## 調整欄位名稱OT為目標欄位
```
data = data.rename(columns={
    data.columns[0]:'date',  data.columns[-1]:'OT'
})
data  = data[['date'] + list(data.columns[2:-1]) + ['OT']]
```
## 儲存處裡完成csv
```
data.to_csv(r'datasets\ETT-data\ETTh1.csv',index=False)
```
 ## 重要參數
```
 --features     type=str,  default='MS'     特徵選取 S:單變量預測單變量(y->y) MS:多變量預測單變量(x+y->y) M:多變量預測多變量(x+y->x+y)(未測試)
 --seq_len      type=int,  default=480      用多少筆資料進行預測 EX: 15min*192 = 2day 
 --label_len    type=int,  default=96       預測標籤長度 EX: 15min*96 = 1day 
 --pred_len     type=int,  default=96       預測長度    EX: 15min*96 = 1day 
 --model'       type=str,  default='SCINet' 模型紀錄名稱
 --model_name   type=str,  default='SCINet' Tensor board 上模型名稱 (啟動方式tensorboard --logdir .\event) http://localhost:6006/
```

## 輔助工具
```
 --evaluate     type=bool, default=False    進行Test data驗證(倒數10%) 結果儲存至 .\exp\ett_results
 --evaluateALL  type=bool, default=False    進行ALL data驗證 結果儲存至 .\exp\ett_results
 --infer        type=bool, default=False    落地實際預測 取.\dataset\infer.csv 倒數seq_len筆 進行預測
```

## 模型配置
```
 --train_epochs type=int,  default=100      訓練迭代次數
 --patience     type=int,  default=15       Valid set 幾次 loss 沒有下降提早終止訓練
 --hidden-size  type=int,  default=2        隱藏層數量 建議1~5
 --batch_size   type=int,  default=8        訓練批次大小 建議2,4,8,16,32,64,128,256
 --lr'          type=float default=3e-3     學習速率建議 1e-3 ~ 1e-5

```

## 訓練Customer Data 請將 run_Etth.py 中 34 修改為當前變量數目(x+y) 
```
data_parser = {
    'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [34, 34, 34], 'S': [1, 1, 1], 'MS': [34, 34, 1]},
}
```

### Training DEMO
```
python .\run_ETTh.py --hidden-size 2 --batch_size 16  --seq_len 192 --label_len 96 --pred_len 96 --features MS --model_name h2_b16_2day_ms 
```

### Test DEMO
```
python .\run_ETTh.py --hidden-size 2 --batch_size 16  --seq_len 192 --label_len 96 --pred_len 96 --features MS --evaluateAll True
```

### Infer Demo
```
python .\run_ETTh.py --hidden-size 2 --batch_size 16  --seq_len 192 --label_len 96 --pred_len 96 --features MS  --infer True
```
