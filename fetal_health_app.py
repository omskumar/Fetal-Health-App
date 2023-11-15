# Import libraries
import streamlit as st
import pandas as pd
import pickle

#title
st.title('Fetal Health Classification: A Machine Learning App') 
# Display the image
st.image('fetal_health_image.gif', width=700)
st.subheader("Utilize my advanced Machine Learning application to predict fetal health classifications.") 

# Reading the pickle files that we created before 
rf_pickle = open('rf_fetal_health.pickle', 'rb') 
rf_model = pickle.load(rf_pickle) 
rf_pickle.close() 

#Show example upload to users
st.write("Example upload - Please follow the same format!")
example_df = pd.read_csv('fetal_health.csv')
example_df = example_df.drop(columns = ['fetal_health'])
st.write(example_df.head(3))

#File Upload
user_file = st.file_uploader("Upload your data!")
if user_file is None:
    st.write("Please upload a file!")
    # Showing additional items

else:
    # Loading data
   user_df = pd.read_csv(user_file) # User provided data
   original_df = pd.read_csv('fetal_health.csv') # Original data to create ML model
   # Dropping null values #test this 
   user_df = user_df.dropna() 
   original_df = original_df.dropna()    
   # Remove output column from original data
   original_df = original_df.drop(columns = ['fetal_health'])
   # Ensure the order of columns in user data is in the same order as that of original data
   user_df = user_df[original_df.columns]
   # Concatenate two dataframes together along rows (axis = 0)
   combined_df = pd.concat([original_df, user_df], axis = 0)
   # Number of rows in original dataframe
   original_rows = original_df.shape[0]
   # Create dummies for the combined dataframe
   combined_df_encoded = pd.get_dummies(combined_df)
   # Split data into original and user dataframes using row index
   original_df_encoded = combined_df_encoded[:original_rows]
   user_df_encoded = combined_df_encoded[original_rows:]
   # Predictions for user data
   user_pred = rf_model.predict(user_df_encoded)
   # Adding predicted class to user dataframe
   user_df['Predicted Fetal Health Class'] = user_pred
   # Prediction Probabilities
   user_pred_prob = rf_model.predict_proba(user_df_encoded)
   # Storing the maximum prob. in a new column
   user_df['Predicted Probability (%)'] = user_pred_prob.max(axis = 1)*100
   st.subheader("Predicting Fetal Health Class")
   #color coding
   def color_class(fetal_class):
    color = 'Lime' if fetal_class=="Normal" else 'Yellow' if fetal_class=='Suspect' else 'Orange'
    return f'background-color: {color}'
   st.dataframe(user_df.style.applymap(color_class, subset=['Predicted Fetal Health Class']))
   
# Showing additional items
st.subheader("Prediction Performance")
tab1, tab2, tab3 = st.tabs(["Feature Importance", 'Confusion Matrix', 'Classification Report'])
with tab1:
   st.image('rf_feature_imp.svg')
with tab2:
   st.image('rf_confusion_matrix.svg')
with tab3:
   class_report_df = pd.read_csv('rf_class_report.csv', index_col = 0)
   st.dataframe(class_report_df)
