import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

from ml import load_and_split_data, train_knn, train_rf, compare_models_mcnemar


st.title('Cervical Cancer (Risk Factors)')
st.header('Imputierte Version des Datensatzes')

df = pd.read_csv('Data/df_mice.csv')
st.dataframe(df)


# ---------------------------------------------------------
# Featureauswahl + Statistiken
# ---------------------------------------------------------
st.header('Featureauswahl')
cols = df.columns
choice = st.selectbox('Pick a column', cols, key='hist_select')

st.header('Statistische Werte')
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric('Mittelwert', f'{df[choice].mean():.3f}')
with col2:
    st.metric('Standardabweichung', f'{df[choice].std():.3f}')
with col3:
    st.metric('Minimum', f'{df[choice].min():.3f}')
with col4:
    st.metric('Maximum', f'{df[choice].max():.3f}')


# ---------------------------------------------------------
# Histogramm
# ---------------------------------------------------------
st.header('Histogramme der numerischen Features')
fig1 = px.histogram(df, x=choice, color='Biopsy', marginal="box", hover_data=df.columns)
st.plotly_chart(fig1, theme="streamlit")


# ---------------------------------------------------------
# ML: Daten laden + Modellwahl
# ---------------------------------------------------------
X_train, X_test, y_train, y_test, X, y = load_and_split_data()

st.header("Machine Learning")

model_choice = st.selectbox("Pick a learner", ["KNN", "RandomForest"])


# ---------------------------------------------------------
# Caching für Modelle
# ---------------------------------------------------------
@st.cache_resource
def get_knn_model(X_train, y_train):
    return train_knn(X_train, y_train)

@st.cache_resource
def get_rf_model(X_train, y_train):
    return train_rf(X_train, y_train)


# ---------------------------------------------------------
# Modelltraining (gecached)
# ---------------------------------------------------------
if model_choice == "KNN":
    best_model, params, score, train_sizes, train_scores, test_scores = get_knn_model(X_train, y_train)
else:
    best_model, params, score, train_sizes, train_scores, test_scores = get_rf_model(X_train, y_train)

st.subheader("Bestes Modell mit GridSearch")

# ---------------------------------------------------------
# Modelle laden (gecached)
# ---------------------------------------------------------
knn_model, _, _, _, _, _ = get_knn_model(X_train, y_train)
rf_model, _, _, _, _, _ = get_rf_model(X_train, y_train)



# ---------------------------------------------------------
# Tabs: Estimator / Hyperparameters
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["Estimator", "Hyperparameters"])

with tab1:
    st.write(best_model)

with tab2:
    if model_choice == "KNN":
        st.json(best_model.named_steps["knn"].get_params())
    else:
        st.json(best_model.get_params())


# ---------------------------------------------------------
# CV Score
# ---------------------------------------------------------
st.write("Best CV score:", score)


# ---------------------------------------------------------
# Learning Curve Plot
# ---------------------------------------------------------
st.header("Learning Curve")

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_std = test_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_sizes, train_mean, 'o-', label='Training Score')
ax.plot(train_sizes, test_mean, 'o-', label='Validation Score')

ax.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.2)
ax.fill_between(train_sizes, test_mean-test_std, test_mean+test_std, alpha=0.2)

ax.set_title(f"Learning Curve - {model_choice}")
ax.set_xlabel("Training Size")
ax.set_ylabel("Accuracy")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# ---------------------------------------------------------
# Shapley wen RF
# ---------------------------------------------------------

st.header('Shapley Values beim RandomForest')
if model_choice == 'RandomForest':
    st.image('images/beeswarm.png')

if model_choice == "RandomForest":
    st.header("Feature Importances")
    st.image('images/feature_importance_rf.png')

# ---------------------------------------------------------
# PCA für KNN
# ---------------------------------------------------------

if model_choice == "KNN":
    st.header("KNN - Datenvisualisierung mit PCA")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train)
    
    fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=y_train,
                     title=f"PCA (erklärt {pca.explained_variance_ratio_.sum():.1%} der Varianz)")
    st.plotly_chart(fig)
    
    st.caption(f"k={best_model.named_steps['knn'].n_neighbors if hasattr(best_model, 'named_steps') else best_model.n_neighbors} Nachbarn")


# ---------------------------------------------------------
# Modelle vergleichen
# ---------------------------------------------------------
st.header("McNemar-Test: KNN vs RandomForest")
result = compare_models_mcnemar(knn_model, rf_model, X_test, y_test)
st.write(result["interpretation"])
st.table(result["contingency"])
