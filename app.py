import streamlit as st
import pickle
from sklearn.preprocessing import MinMaxScaler
import joblib
st.title("Singapore Resale Flat Price")

st.markdown(
    """<div style="text-align: right; margin-right: 50px;">
        <h3 style="color: #1E90FF;">- Created by Gokulraj Pandiyarajan</h3>
    </div>""",
    unsafe_allow_html=True
)
st.write()
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
month=st.selectbox('Select Month',months)
month_number={'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7, 'August':8, 'September':9 , 'October':10, 'November':11, 'December':12}
month=month_number[month]
st.write(month)

year=st.selectbox('Select Year',[x for x in range(1990, 2023+1) ])
st.write(year)
town_list=['ANG MO KIO','BEDOK','BISHAN','BUKIT BATOK','BUKIT MERAH','BUKIT PANJANG','BUKIT TIMAH','CENTRAL AREA','CHOA CHU KANG','CLEMENTI','GEYLANG','HOUGANG','JURONG EAST','JURONG WEST','KALLANG/WHAMPOA','LIM CHU KANG','MARINE PARADE','PASIR RIS','PUNGGOL','QUEENSTOWN','SEMBAWANG','SENGKANG','SERANGOON','TAMPINES','TOA PAYOH','WOODLANDS','YISHUN']
town=st.selectbox("Select Town",town_list)
for numbers, towns in enumerate(town_list):
    if town==towns:
        town=numbers    
st.write(town)

flat_type_list=['1 ROOM','2 ROOM','3 ROOM','4 ROOM','5 ROOM','EXECUTIVE','MULTI-GENERATION']
flat_type=st.selectbox("Select a Flat Type",flat_type_list)
for number,flat in enumerate(flat_type_list):
    if flat_type==flat:
        flat_type=number

st.write(flat_type)

storey_range_list=['01 TO 03','01 TO 05','04 TO 06','06 TO 10','07 TO 09','10 TO 12','11 TO 15','13 TO 15','16 TO 18','16 TO 20','19 TO 21','21 TO 25','22 TO 24','25 TO 27','26 TO 30','28 TO 30','31 TO 33','31 TO 35','34 TO 36','36 TO 40','37 TO 39','40 TO 42','43 TO 45','46 TO 48','49 TO 51']
storey_range=st.selectbox("Select a Storey Range",storey_range_list)
for number, storey in enumerate(storey_range_list):
    if storey_range == storey:
        storey_range=number
st.write(storey_range)

floor_area_sqm= st.number_input("Floor Area Sqm ", min_value=28, max_value=307, step=1)
st.write(floor_area_sqm)

flat_model_list=['2-ROOM','3Gen','APARTMENT','Adjoined flat','DBSS','IMPROVED','IMPROVED-MAISONETTE','MAISONETTE','MODEL A','MODEL A-MAISONETTE','MODEL A2','MULTI GENERATION','NEW GENERATION','PREMIUM APARTMENT','PREMIUM APARTMENT Loft','Premium MAISONETTE','SIMPLIFIED','STANDARD','TERRACE','Type S1','Type S2']
flat_model=st.selectbox("Choose Flat Model", flat_model_list)
for number, flat_models in enumerate(flat_model_list):
    if flat_models==flat_model:
        flat_model=number

st.write(flat_model)

lease_commence_date=st.selectbox('Select Lease Commence Date',[x for x in range(1966, 2023+1) ])
st.write(lease_commence_date)

predict=st.button("Predict")
if predict:
    x=[[month, year,town,flat_type,storey_range,floor_area_sqm,flat_model,lease_commence_date]]
    print(x)    
    # Load the scaler from the file
    scaler = joblib.load('mm.pkl')

    # Now you can use this loaded scaler to transform new data
    x_new = scaler.transform(x)
    print(x_new)

    with open("random_forest.pkl", "rb") as file:
        rf=pickle.load(file)#

    rf_preds=rf.predict(x_new)
    print(rf_preds)

    st.title(rf_preds[0])
