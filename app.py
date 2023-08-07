import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Function to create a simple ML pipeline and perform predictions
def predict_prices(df):
    # Load the saved pipeline
    df['manufacturer']=df['name'].apply(lambda x:x.split(' ')[0])
    df['model']=df['name'].apply(lambda x:x.split(' ')[1])
    df['No_of_total_years']=2023-df['year']
    df.drop(['name','year'],axis=1,inplace=True) 
    with open(r'pipeline.pkl', 'rb') as file:
        pipeline = pickle.load(file)


    # Make predictions
    predictions = pipeline.predict(df)

    df['predicted_selling_price'] = predictions
    return df['predicted_selling_price']

def main():
    st.title("Car Details Input and Prediction")
    st.header("Enter Car Details")

    # Input fields for each column
    name = st.text_input("Name")
    year = st.number_input("Year", min_value=1980, max_value=2023, step=1)
    km_driven = st.number_input("Kilometers Driven", min_value=0, step=1000)
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])

    # Display the input data in a DataFrame
    data = {
        "name": name,
        "year": year,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner
    }
    df = pd.DataFrame(data,index=[0])

    # Prediction button
    if st.button("Predict"):
        # Call the predict_prices function to perform predictions
        df_with_predictions = predict_prices(df)

        # Display the DataFrame with predicted selling prices
        st.header("Car Details with Predicted Selling Prices")
        st.dataframe(df_with_predictions)

if __name__ == "__main__":
    # Train the model and save the pipeline (you can do this in a separate script)
    #train_and_save_pipeline()

    # Run the Streamlit app
    main()
