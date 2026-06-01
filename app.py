import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------------------------------
# 1. Page Configuration
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="wide"
)

st.title("Credit Risk Prediction Dashboard")
st.markdown("Enter the customer's financial and background details below.")

# -------------------------------------------------------------------------
# 2. Model Loading
# -------------------------------------------------------------------------
# Load the trained CatBoost model
@st.cache_resource
def load_model():
    model = joblib.load("models/cat_best_model.joblib") 
    return model

try:
    model = load_model()
except Exception as e:
    st.warning("Model file not found. Please ensure it is in the correct directory.")
    model = None

# -------------------------------------------------------------------------
# 3. User Input Section
# -------------------------------------------------------------------------
# Mappings for categorical variables
home_values = {1: 'rent', 2: 'owner', 3: 'private', 4: 'ignore', 5: 'parents', 6: 'other', 0: 'unk'}
marital_values = {1: 'single', 2: 'married', 3: 'widow', 4: 'separated', 5: 'divorced', 0: 'unk'}
records_values = {1: 'no', 2: 'yes', 0: 'unk'}
job_values = {1: 'fixed', 2: 'partime', 3: 'freelance', 4: 'others', 0: 'unk'}

st.subheader("Customer Features")

# Use columns to distribute inputs and prevent excessive vertical scrolling
col1, col2, col3 = st.columns(3)

with col1:
    seniority = st.number_input("Seniority (Years)", min_value=0, max_value=50, value=2)
    time = st.number_input("Time (Months)", min_value=1, max_value=120, value=36)
    age = st.number_input("Age", min_value=10, max_value=100, value=30)
    expenses = st.number_input("Expenses", min_value=0.0, value=50.0)
    income = st.number_input("Income", min_value=0.0, value=100.0)

with col2:
    assets = st.number_input("Assets", min_value=0.0, value=2000.0)
    debt = st.number_input("Debt", min_value=0.0, value=0.0)
    amount = st.number_input("Amount", min_value=0.0, value=500.0)
    price = st.number_input("Price", min_value=0.0, value=600.0)

with col3:
    home = st.selectbox("Home", options=list(home_values.keys()), format_func=lambda x: home_values[x])
    marital = st.selectbox("Marital", options=list(marital_values.keys()), format_func=lambda x: marital_values[x])
    records = st.selectbox("Records", options=list(records_values.keys()), format_func=lambda x: records_values[x])
    job = st.selectbox("Job", options=list(job_values.keys()), format_func=lambda x: job_values[x])

# Combine inputs into a DataFrame
# Note: Storing the mapped string values for categorical features
data = {
    'seniority': seniority,
    'home': home_values[home],
    'time': time,
    'age': age,
    'marital': marital_values[marital],
    'records': records_values[records],
    'job': job_values[job],
    'expenses': expenses,
    'income': income,
    'assets': assets,
    'debt': debt,
    'amount': amount,
    'price': price
}

input_df = pd.DataFrame([data])

# Feature Engineering (5 Engineered Features)
input_df['diff'] = input_df['price'] - input_df['amount']
input_df['ratio'] = input_df['price'] / (input_df['amount'] + 1)
input_df['debt_income_ratio'] = input_df['debt'] / (input_df['income'] + 1)
input_df['assets_amount_ratio'] = input_df['assets'] / (input_df['amount'] + 1)

# Create 'age_group' and drop original 'Age'
bins = list(range(10, 80, 10))
input_df['age_group'] = pd.cut(input_df['age'], bins=bins).astype(str)
input_df = input_df.drop(columns=['age'])


# -------------------------------------------------------------------------
# 4. Make Prediction Section
# -------------------------------------------------------------------------

st.subheader("Data Overview")
st.dataframe(input_df)

if st.button("Predict Credit Risk"):
    if model is not None:
        pred_df = input_df.copy()
        
        # Ensure categorical features are explicitly cast to string for CatBoost
        cat_cols = ['home', 'marital', 'records', 'job', 'age_group'] 
        for col in cat_cols:
            if col in pred_df.columns:
                pred_df[col] = pred_df[col].astype(str)
        
        # --- THE FIX: Reorder columns to match the trained model exactly ---
        try:
            expected_cols = model.feature_names_
            pred_df = pred_df[expected_cols]
        except KeyError as e:
            st.error(f"Column mismatch error. The model expects different columns: {e}")
            st.stop()
        # -------------------------------------------------------------------
        
        # Make prediction
        prediction = model.predict(pred_df)[0]
        prediction_proba = model.predict_proba(pred_df)[0][1] 
        
        st.markdown("---")
        st.subheader("Prediction Result")
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if prediction == 1:
                st.error("High Risk (Default)")
            else:
                st.success("Low Risk (OK)")
                
        with res_col2:
            st.metric(label="Probability of Default", value=f"{prediction_proba:.2%}")
    else:
        st.error("Model not found. Cannot perform prediction.")