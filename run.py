import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import altair as alt
from subprocess import Popen, PIPE, CalledProcessError
import time,sys
from datetime import datetime,timedelta
from glob import glob
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from PIL import Image
import dateutil.parser


def rmse(true,pred):
    return mean_squared_error(true.flatten(),pred.mean(axis=2).flatten())**0.5

    
def fig_plotly(time,target,n=0):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x= data[time], y=data[target],line=dict( width=2)))
    fig.add_vrect(
        x0=data[time].iloc[int(data.shape[0]*0.8-n)],
        x1=data[time].iloc[int(data.shape[0]*0.9-n)],
        fillcolor="red",
        opacity=0.2,
        line_width=0,
    )

    fig.add_vrect(
        x0=data[time].iloc[int(data.shape[0]*0.9)-n],
        x1=data[time].iloc[-1-n],
        fillcolor="green",
        opacity=0.2,
        line_width=0,
    )
    return fig

def execute_experiment(cmd):

    with st.empty():
        with Popen(cmd, stdout=PIPE, bufsize=1, universal_newlines=True) as process:
            for line in process.stdout:
                if line.find('cost time') >-1:
                    st.write(f"⏳ {line}")
                    p = float(line.split(' cost')[0].split(': ')[-1].replace('\n',''))
                    my_bar.progress(int((p/train_epochs)*100))

def color_survived(val):
    color = 'green' if val else 'red'
    return f'background-color: {color}'


st.markdown(rf'''
# SCINet 時間序列預測套件
Time Series is a Special Sequence: Forecasting with Sample Convolution and Interaction''')

st.image(Image.open('./arXivsvg.png'))
st.image(Image.open('./PyTorch.png'))

tab1, tab2, tab3 = st.tabs(["📈 data process", "🗃 model parameters", "🖼️ Evaluate Result"])
with tab1:
    st.header('選擇上傳CSV檔案')
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:

        # Can be used wherever a "file-like" object is accepted:
        data = pd.read_csv(uploaded_file)
        cols = list(data.columns)

        st.subheader('時間欄位')
        col_time = st.selectbox(
            'col_time',
            cols,
            index=0
            )
        st.subheader('預測目標')
        col_target = st.selectbox(
            'col_target',
            cols,
            index=data.shape[1]-1
            )

        st.write(data.head())

        
        TIME = list(map(lambda x : dateutil.parser.parse(x),data[col_time]))
        data[col_time] = TIME

        st.subheader('選擇資料範圍')
        appointment = st.slider(
            "appointment",
            min_value = TIME[0],
            max_value = TIME[-1],
            step = TIME[1] - TIME[0],
            value=(TIME[0], TIME[-1]))

        st.caption(f'''開始時間 {appointment[0].strftime("%x")} {appointment[0].strftime("%X")}   ''')
        st.caption(f'''結束時間 {appointment[-1].strftime("%x")} {appointment[-1].strftime("%X")}''')

        data = data[(data[col_time] > appointment[0])]
        data = data[(data[col_time] < appointment[-1])]

        fig = fig_plotly(col_time,col_target,n=0)

        st.plotly_chart(fig, use_container_width=True)
        
        if st.button('確認完成'):

            #丟棄NA與std=0欄位
            data = data.dropna(axis=0).drop(data.std()[(data.std() == 0)].index, axis=1)

            #調整欄位名稱OT為目標欄位
            data = data.rename(columns={
                col_time:'date',  col_target:'OT'
            })
            cols = list(data.columns)
            cols.remove('date')
            cols.remove('OT')

            data  = data[['date'] + cols + ['OT']]

            #儲存處裡完成csv
            data.to_csv(r'datasets\ETT-data\ETTh1.csv',index=False)
            st.caption('儲存處裡完成CSV :sunglasses: ')



    # 重要參數
    # --features     type=str,  default='MS'     特徵選取 S:單變量預測單變量(y->y) MS:多變量預測單變量(x+y->y) M:多變量預測多變量(x+y->x+y)(未測試)
    # --seq_len      type=int,  default=480      用多少筆資料進行預測 EX: 15min*192 = 2day 
    # --label_len    type=int,  default=96       預測標籤長度 EX: 15min*96 = 1day 
    # --pred_len     type=int,  default=96       預測長度    EX: 15min*96 = 1day 
    # --model'       type=str,  default='SCINet' 模型紀錄名稱
    # --model_name   type=str,  default='SCINet' Tensor board 上模型名稱 (啟動方式tensorboard --logdir .\event) http://localhost:6006/

    # 輔助工具
    # --evaluate     type=bool, default=False    進行Test data驗證(倒數10%) 結果儲存至 .\exp\ett_results
    # --evaluateALL  type=bool, default=False    進行ALL data驗證 結果儲存至 .\exp\ett_results
    # --infer        type=bool, default=False    落地實際預測 取.\dataset\infer.csv 倒數seq_len筆 進行預測

    # 模型配置
    # --train_epochs type=int,  default=100      訓練迭代次數
    # --patience     type=int,  default=15       Valid set 幾次 loss 沒有下降提早終止訓練
    # --hidden-size  type=int,  default=2        隱藏層數量 建議1~5
    # --batch_size   type=int,  default=8        訓練批次大小 建議2,4,8,16,32,64,128,256
    # --lr'          type=float default=3e-3     學習速率建議 1e-3 ~ 1e-5
with tab2:
    st.header('設置模型超參數')
    st.subheader('特徵選取')
    features = st.radio(
    "features",
    ('S', 'MS', 'M(測試中)'))
    st.caption('S:單變量預測單變量(y->y)')
    st.caption('MS:多變量預測單變量(x+y->y)')
    st.caption('M:多變量預測多變量(x+y->x+y)(未測試)')

    st.subheader('一次預測往回看多少筆資料')
    seq_len = st.number_input('seq_len', value=192, step=4)
    st.caption('EX: 15min*192 = 2day')
    if seq_len /8 %1 != 0:
        st.error('seq_len須為8的倍數')

    st.subheader('一次預測未來多少筆資料')
    label_len = st.number_input('label_len', value=96, step=2)
    pred_len = label_len
    st.caption('EX: 15min*96 = 1day')
    
    st.subheader('模型紀錄名稱')
    model_name = st.text_input('model_name', value='SCINET')
    model = model_name
    st.write('模型名稱:', model_name)

    st.subheader('訓練迭代次數')
    train_epochs = st.number_input('train_epochs', value=10)

    st.subheader('驗證及損失函數幾次沒有下降提早終止訓練')
    patience = st.number_input('patience', value=15)

    st.subheader('隱藏層數量')
    hidden_size = st.slider('hidden_size', min_value=1, max_value=5, value=1)

    st.subheader('訓練批次大小')
    batch_size = st.selectbox('batch_size',('2','4','8','16','32','64','128','256'),index=3)

    st.subheader('學習速率')
    lr = st.selectbox('lr',('1e-3', '1e-4', '1e-5'),index=0)

    code = f'''python {os.getcwd()}\\run_ETTh.py --features {features} --seq_len {seq_len} --label_len {label_len} --pred_len {label_len} --model {model_name} --model_name {model_name} --train_epochs {train_epochs} --patience {patience} --hidden-size {hidden_size} --batch_size {batch_size}  --lr {lr}'''
    st.code(code, language='python')

    if st.button('開始訓練'):
        my_bar = st.progress(0)
        st.write(f"📝 start training please wait")
        seconds = execute_experiment(code)
        st.write(f"🤗 Training Finish")
        st.write(f"⏳ Start Evaluate All Data")
        seconds = execute_experiment(code + ' --evaluateAll True')
        st.write(f"🤗 Finish All Task")

with tab3:
    files = glob(r'exp\ett_results\*')
    st.subheader('選擇結果')
    f = st.selectbox('select result',files,index=0)
    true = np.load(rf'{f}\true_scales.npy')
    pred = np.load(rf'{f}\pred_scales.npy')
    x = st.slider('x', min_value=1, max_value=true.shape[0]-1, value=1)
    L = true.shape[0]
    col1, col2, col3 = st.columns(3)
    train_rmse = round(rmse(true[0:int(L*0.8)],pred[0:int(L*0.8)]),2)
    valid_rmse = round(rmse(true[int(L*0.8):int(L*0.9)],pred[int(L*0.8):int(L*0.9)]),2)
    test_rmse  = round(rmse(true[int(L*0.9):-1],pred[int(L*0.9):-1]),2)
    col1.metric("TRAIN RMSE", train_rmse)
    col2.metric("VALID RMSE", valid_rmse)
    col3.metric("TEST  RMSE", test_rmse)
    # st.write('TRAIN RMSE:{:.2f}'.format(rmse(true[0:int(L*0.8)],pred[0:int(L*0.8)])))
    # st.write('VALID RMSE:{:.2f}'.format(rmse(true[int(L*0.8):int(L*0.9)],pred[int(L*0.8):int(L*0.9)])))
    # st.write('TEST  RMSE:{:.2f}'.format(rmse(true[int(L*0.9):-1],pred[int(L*0.9):-1])))
    def show(x=(0,true.shape[0]-1,1)):
        # fig = plt.figure(figsize=(20,5),dpi=1500)
        if x > true.shape[0]*0.8:
            title = 'Valid'
            if x > true.shape[0]*0.9:
                title = 'Test'
        else:
            title = 'Train'
        # plt.ylim([true.min(), true.max()])
        # plt.plot(true[x].flatten())
        # plt.plot(pred[x].mean(axis=1).flatten())
        # plt.show()
        fig = go.Figure()

        fig.add_trace(go.Scatter(y=true[x].flatten(),line=dict( width=2),name="true"))
        fig.add_trace(go.Scatter(y=pred[x].mean(axis=1).flatten(),line=dict( width=1),name="pred"))
        fig.update_layout(
            title=title,
            xaxis_title="X Axis Title",
            yaxis_title="Y Axis Title",
            font=dict(
                size=18,
            )
        )
        return fig

    st.plotly_chart(show(x), use_container_width=True)