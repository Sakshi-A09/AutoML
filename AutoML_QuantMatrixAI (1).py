#!/usr/bin/env python
# coding: utf-8

#%%writefile app.py
import streamlit as st
import h2o
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from h2o.automl import H2OAutoML
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from scipy.stats import zscore
from statsmodels.tsa.api import detrend

h2o.init(max_mem_size="8G")

st.title("H2O AutoML with EDA and Feature Engineering")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df.loc[:, ~df.columns.duplicated()]
    df.columns = df.columns.str.replace(" ", "_")
    df.dropna(inplace=True)
    
    df = df.dropna(subset=["Volume"])  
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df.dropna(subset=["Volume"], inplace=True)
    df = df[df["Volume"] > 0]  

    st.write("### Filter Data Using L0 and L1")
    col_L0 = st.selectbox("Select Feature for L0 Filter", ['None'] + list(df.columns), key="L0_feature")
    if col_L0 != 'None':
        unique_vals_L0 = ['None'] + sorted(df[col_L0].dropna().unique())
        val_L0 = st.selectbox(f"Select Value from {col_L0}", unique_vals_L0, key="L0_value")
        if val_L0 != 'None':
            df = df[df[col_L0] == val_L0]

    col_L1 = st.selectbox("Select Feature for L1 Filter", ['None'] + list(df.columns), key="L1_feature")
    if col_L1 != 'None':
        unique_vals_L1 = ['None'] + sorted(df[col_L1].dropna().unique())
        val_L1 = st.selectbox(f"Select Value from {col_L1}", unique_vals_L1, key="L1_value")
        if val_L1 != 'None':
            df = df[df[col_L1] == val_L1]

    st.write("Filtered Dataset", df.head())
    st.write("Dataset shape before AutoML:", df.shape)
    
    drop_cols = ["D2", "D3", "D4", "D5", "D6", "AV1", "AV2", "AV3", "AV4", "AV5", "AV6", "EV1", "EV2", "EV3", "EV4", "EV5", "EV6"]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df.dropna(subset=["Date"], inplace=True)

    if "SalesValue" in df.columns and "Volume" in df.columns:
        if "ListPrice" in df.columns:
            df["Price"] = df["ListPrice"]
        else:
            raw_price = df["SalesValue"] / df["Volume"]
            df["Price"] = detrend(raw_price.values)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(subset=["Price"], inplace=True)
        df.drop(columns=["SalesValue"], inplace=True, errors="ignore")


    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week
    df["Year"] = df["Date"].dt.year
    df["Quarter"] = df["Date"].dt.quarter

    st.subheader("Target and Feature Selection")
    y_variable = st.selectbox("Select Y-variable (Target)", options=["None"] + list(df.columns), index=0)
    if y_variable == "None":
        st.stop()

    x_variables = st.multiselect("Select X-variables (Features)", options=[col for col in df.columns if col != y_variable])
    
#     st.subheader("Target and Feature Selection")
#     y_variable = st.selectbox("Select Y-variable (Target)", options=["None"] + list(df.columns), index=0)
#     if y_variable == "None":
#         st.stop()

#     x_variables = st.multiselect("Select X-variables (Features)", options=[col for col in df.columns if col != y_variable])

    
    st.write("### Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(15, 8))  
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax, annot_kws={"size": 10})
    st.pyplot(fig)
    
    st.subheader("Price vs Target Relationship")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Price", y=df.columns[df.columns.get_loc("Price")+1], ax=ax)  # Plot next column after Price
    st.pyplot(fig)
    


    if "Date" in df.columns:
        st.subheader("Time Series Plots")
        with st.expander("Optional Time Filters (Year / Month)", expanded=False):
            years = sorted(df["Year"].dropna().unique())
            months = sorted(df["Month"].dropna().unique())
            selected_year = st.selectbox("Select Year", options=["All"] + list(years), index=0)
            selected_month = st.selectbox("Select Month", options=["All"] + list(months), index=0)

        ts_df = df.copy()
        if selected_year != "All":
            ts_df = ts_df[ts_df["Year"] == selected_year]
        if selected_month != "All":
            ts_df = ts_df[ts_df["Month"] == selected_month]

        grouped_ts = ts_df.groupby("Date")[[y_variable] + x_variables].mean(numeric_only=True).reset_index()

        for feature in x_variables:
            if feature not in grouped_ts.columns:
                st.warning(f"Skipping feature '{feature}' — not in grouped data.")
                continue
            if not pd.api.types.is_numeric_dtype(grouped_ts[feature]):
                continue

            fig, ax1 = plt.subplots(figsize=(10, 4))
            ax1.set_xlabel("Date")
            ax1.set_ylabel(y_variable, color="tab:red")
            ax1.plot(grouped_ts["Date"], grouped_ts[y_variable], color="tab:red")
            ax1.tick_params(axis='y', labelcolor="tab:red")

            ax2 = ax1.twinx()
            ax2.set_ylabel(feature, color="tab:blue")
            ax2.plot(grouped_ts["Date"], grouped_ts[feature], color="tab:blue")
            ax2.tick_params(axis='y', labelcolor="tab:blue")

            fig.tight_layout()
            st.pyplot(fig)


    st.subheader("Outlier Removal")
    outlier_method = st.selectbox("Choose method", ["None", "Z-Score", "IQR"])
    initial_size = df.shape[0]

    if outlier_method != "None":
        target_series = df[y_variable].astype(float)
        if outlier_method == "Z-Score":
            z_scores = np.abs(zscore(target_series))
            df = df[z_scores < 3]
        elif outlier_method == "IQR":
            Q1, Q3 = target_series.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            df = df[(target_series >= Q1 - 1.5 * IQR) & (target_series <= Q3 + 1.5 * IQR)]
        st.write(f"Outliers removed: {initial_size - df.shape[0]}")

    st.subheader("Feature Transformations")
    feature_transforms = {}
    for feature in x_variables:
        if pd.api.types.is_numeric_dtype(df[feature]):

            if feature == 'Price':
                st.warning(f"Skipping transforms for Price (monotonic constraint)")
                continue
            transform = st.selectbox(f"Transform for {feature}", ["None", "Log", "Power", "Standardize"], key=f"trans_{feature}")
            feature_transforms[feature] = transform

    for feature, method in feature_transforms.items():
        try:
            if method == "Log":
                df[feature] = np.log1p(df[feature])
            elif method == "Power":
                df[feature] = np.power(df[feature], 0.5)
            elif method == "Standardize":
                df[feature] = (df[feature] - df[feature].mean()) / df[feature].std()
        except Exception as e:
            st.warning(f"Could not transform {feature}: {e}")


    monotone_constraints = {'Price': -1} if 'Price' in x_variables else None
    
    df["Date"] = df["Date"].astype(str)
    model_cols = list(dict.fromkeys([y_variable] + x_variables + ["Date"]))
    df_h2o = h2o.H2OFrame(df[model_cols])

    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col != "Date" and col in df_h2o.columns:
            df_h2o[col] = df_h2o[col].asfactor()

    x_vars = [col for col in df_h2o.columns if col != y_variable]
    train, test = df_h2o.split_frame(ratios=[0.8], seed=42)
    target = y_variable

    st.write(f"Training rows: {train.nrows}, Testing rows: {test.nrows}")

    if st.button("Run AutoML"):
        aml = H2OAutoML(
            max_runtime_secs=1200,
            include_algos=["GBM", "XGBoost"],
            monotone_constraints=monotone_constraints,
            seed=42
        )
        aml.train(x=x_vars, y=y_variable, training_frame=train)
        

        top_models = []

        def update_top_models(model_id):
            model = h2o.get_model(model_id)
            preds = model.predict(test).as_data_frame().values.flatten()
            actuals = test[target].as_data_frame().values.flatten()

            mask = ~np.isnan(preds) & ~np.isnan(actuals)
            preds, actuals = preds[mask], actuals[mask]

            if len(preds) == 0:
                return

            rmse = np.sqrt(mean_squared_error(actuals, preds))
            mape = mean_absolute_percentage_error(actuals, preds)

            valid_mask = (actuals >= 0) & (preds >= 0)
            rmsle = np.sqrt(mean_squared_error(np.log1p(actuals[valid_mask]), np.log1p(preds[valid_mask]))) if valid_mask.sum() > 0 else np.nan

            model_info = {
                "model_id": model_id,
                "rmse": rmse,
                "mse": mean_squared_error(actuals, preds),
                "mae": np.mean(np.abs(actuals - preds)),
                "rmsle": rmsle,
                "mean_residual_deviance": np.mean((actuals - preds) ** 2),
                "mape": mape
            }

            top_models.append(model_info)
            top_models.sort(key=lambda x: x["rmse"])
            top_models[:] = top_models[:5]

        leaderboard_df = aml.leaderboard.as_data_frame()
        if "model_id" in leaderboard_df.columns:
            for model_id in leaderboard_df["model_id"]:
                update_top_models(model_id)
        else:
            st.warning("No models found in leaderboard.")
    
        top_models_df = pd.DataFrame(top_models)
        st.write("### AutoML Top 5 Models")
        st.dataframe(top_models_df)

        if len(top_models) > 0:
            overall_rmse = np.mean([m["rmse"] for m in top_models])
            overall_mape = np.mean([m["mape"] for m in top_models])
            st.write(f"*Overall RMSE:* {overall_rmse}")
            st.write(f"*Overall MAPE:* {overall_mape}")
            

        shap_values = {}
        model_preds = {}

        X_test_df = test.drop(y_variable).as_data_frame()
        y_test = test[y_variable].as_data_frame().values.flatten()

        for model_info in top_models:
            model_id = model_info["model_id"]
            model = h2o.get_model(model_id)

            try:
                if model.algo not in ["gbm", "xgboost"]:
                    raise ValueError(f"Unsupported algorithm: {model.algo}")

                contrib = model.predict_contributions(test).as_data_frame()
                bias = contrib["BiasTerm"]
                contrib = contrib.drop("BiasTerm", axis=1)
                contrib = contrib.reindex(columns=X_test_df.columns)
                

                if 'Price' in contrib.columns and monotone_constraints and monotone_constraints.get('Price') == -1:
                    contrib["Price"] = -np.abs(contrib["Price"])
                
                shap_values[model_id] = contrib.mean().to_frame(name=model_id).T  
                model_preds[model_id] = model.predict(test).as_data_frame().values.flatten()

            except Exception as e:
                st.warning(f"Failed SHAP for {model_id}: {e}")

        if shap_values:
            shap_grid_df = pd.concat(shap_values.values(), keys=shap_values.keys()).T
            shap_grid_df.columns.name = None
            st.subheader("SHAP Values Grid (mean contributions)")
            st.dataframe(shap_grid_df.style.format("{:.4f}"), use_container_width=True)


        elasticity_results = {}
        for model_id, shap_df in shap_values.items():
            try:
                shap_df = shap_df.T.squeeze() if isinstance(shap_df, pd.DataFrame) else shap_df
                mean_prediction = np.abs(np.mean(model_preds[model_id])) 
                elasticities = shap_df / mean_prediction
                elasticity_results[model_id] = elasticities
            except Exception as e:
                st.warning(f"Elasticity calculation failed for {model_id}: {e}")

        if elasticity_results:
            elasticity_df = pd.DataFrame(elasticity_results)
            st.subheader("Elasticity Grid Across Models")
            st.dataframe(elasticity_df.style.format("{:.5f}"), use_container_width=True)

            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(elasticity_df, annot=True, cmap="coolwarm", center=0, fmt=".2f")
            plt.title("Elasticities by Feature and Model")
            st.pyplot(fig)


        if 'Price' in x_variables:
            st.subheader("Partial Dependence Plot for Price")
            try:
                fig = plt.figure(figsize=(10, 4))
                h2o.partial_plot(aml.leader, train, cols=["Price"], plot=True)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not generate PDP: {e}")


        st.subheader("Actual vs Predicted")
        for model_id, preds in model_preds.items():
            fig, ax = plt.subplots()
            ax.plot(y_test, label="Actual", color="blue")
            ax.plot(preds, label="Predicted", color="red")
            ax.set_title(f"Actual vs Predicted for {model_id}")
            ax.legend()
            st.pyplot(fig)
#!streamlit run app.py
