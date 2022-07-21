from datetime import datetime
import streamlit as st
import pandas as pd
import dateutil.parser

data = pd.read_csv(r'datasets\ETT-data\TrainOneDay.csv')
data.head()
data['Dtime'] = list(map(lambda x : dateutil.parser.parse(x),data['Dtime']))
from datetime import time
appointment = st.slider(
     "Schedule your appointment:",
     value=(time(11, 30), time(12, 45)))
st.write("You're scheduled for:", appointment)