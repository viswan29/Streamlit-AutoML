#Core Pkgs
import os 
import streamlit as st

#EDA Pkgs
import pandas as pd
import numpy as np 

#Viz Pkgs
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

#ML Pkgs
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def main():
    st.title("Auto ML Explorer")
    st.subheader("Simple Data Science Automation with Streamlit")
    
    activities = ["EDA","Plot","Model Building","About"]
    choice = st.sidebar.selectbox("Select activity",activities)

    if choice == "EDA":
        st.subheader("Exploratry Data Analysis")
        
        data = st.file_uploader("Upload Dataset",type=["csv","txt"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            
            if st.checkbox("Show Shape"):
                st.write(df.shape)
            
            if st.checkbox("Show Summary"):
                st.write(df.describe())
                
            if st.checkbox("Show Columns"):
                st.write(df.columns)
                
            if st.checkbox("selected columns to show"):
                selected_columns = st.multiselect("select columns",df.columns)
                new_df = df[selected_columns]
                st.dataframe(new_df)
                
            if st.checkbox("Show Value Counts"):
                st.write(df.iloc[:,-1].value_counts())
                
    elif choice == "Plot":
        st.subheader("Data Visualization")
        
        data = st.file_uploader("Upload Dataset",type=["csv","txt"])
        
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            
            if st.checkbox("Correlation with Seaborn"):
                st.write(sns.heatmap(df.corr(),annot=True))
                st.pyplot()
            
            if st.checkbox("Pie Chart"):
                columns_to_plot = st.selectbox("select 1 column",df.columns)
                pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
                st.write(pie_plot)
                st.pyplot()
            
            all_columns_names = df.columns.to_list()
            type_of_plot = st.selectbox("select Type of Plot",["area","bar","hist","kde","box","line"])
            selected_column_names = st.multiselect("select columns to plot",all_columns_names)
        
            if st.button("Generate Plot"):
                st.success("Generating plot of {} for {}".format(type_of_plot,selected_column_names))
                
                if type_of_plot == 'area':
                    cust_data = df[selected_column_names]
                    st.area_chart(cust_data)
                    
                elif type_of_plot == 'bar':
                    cust_data = df[selected_column_names]
                    st.bar_chart(cust_data)
                    
                elif type_of_plot == 'line':
                    cust_data = df[selected_column_names]
                    st.line_chart(cust_data)
                    
                elif type_of_plot:
                    cust_plot = df[selected_column_names].plot(kind=type_of_plot)
                    st.write(cust_plot)
                    st.pyplot()
                
                
    elif choice == "Model Building":
        st.subheader("Building ML Model")
        
        data = st.file_uploader("Upload Dataset",type=["csv","txt"])
        
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            
            #Model Building
            X = df.iloc[:,0:-1]
            Y = df.iloc[:,-1]
            seed = 7
            
            models = []
            models.append(("LR",LogisticRegression()))
            models.append(("LDA",LinearDiscriminantAnalysis()))
            models.append(("KNN",KNeighborsClassifier()))
            models.append(("CART",DecisionTreeClassifier()))
            models.append(("NB",GaussianNB()))
            models.append(("SVM",SVC()))

            model_names = []
            model_mean = [] 
            model_std = [] 
            all_models = []
            scoring = "accuracy"
            for name,model in models:
                kfold = model_selection.KFold(n_splits=10,random_state=seed)
                cv_results = model_selection.cross_val_score(model,X,Y,cv=kfold,scoring=scoring)
                model_names.append(name)
                model_mean.append(cv_results.mean())
                model_std.append(cv_results.std())
                
                accuracy_results = {"model_name":name,"model_accuracy":cv_results.mean(),"standard Deviation":cv_results.std()}
                all_models.append(accuracy_results)
            
            if st.checkbox("Metrics as a Table"):
                st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_std),columns=["Model Name","Model Accuracy","Std Deviation"]))
                
            if st.checkbox("Metrics as a JSON"):
                st.json(all_models)
                
            
    elif choice == "About":
        st.subheader("About")
        st.text("This Project is for Semi Automation of ML models\nand it's respective features covering Exploratory Data Analysis(EDA),\nData Visualization and Model Building")
        
    
if __name__ == '__main__':
    main()