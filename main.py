import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

from ml import load_and_split_data, train_knn, train_rf, train_rf_with_rfecv, compare_models_mcnemar, compare_models_mcnemar_with_rfecv


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
fig1 = px.histogram(df, x=choice, color='Biopsy', marginal='box', hover_data=df.columns)
st.plotly_chart(fig1, theme='streamlit')


# ---------------------------------------------------------
# ML: Daten laden + Modellwahl
# ---------------------------------------------------------
X_train, X_test, y_train, y_test, X, y = load_and_split_data()

st.header('Machine Learning')

model_choice = st.selectbox('Pick a learner', ['KNN', 'RandomForest', 'RandomForest mit RFECV'])


# ---------------------------------------------------------
# Caching für Modelle
# ---------------------------------------------------------
@st.cache_resource
def get_knn_model(X_train, y_train):
    return train_knn(X_train, y_train)

@st.cache_resource
def get_rf_model(X_train, y_train):
    return train_rf(X_train, y_train)

@st.cache_resource
def get_rfecv_model(X_train, y_train):
    return train_rf_with_rfecv(X_train, y_train)


# ---------------------------------------------------------
# Modelltraining (gecached)
# ---------------------------------------------------------
if model_choice == 'KNN':
    best_model, params, score, train_sizes, train_scores, test_scores = get_knn_model(X_train, y_train)
elif model_choice == 'RandomForest':
    best_model, params, score, train_sizes, train_scores, test_scores = get_rf_model(X_train, y_train)
else:
    best_model, params, score, train_sizes, train_scores, test_scores, rfecv, X_train_selected = get_rfecv_model(X_train, y_train)
    X_test_selected = rfecv.transform(X_test)

st.subheader('Bestes Modell mit GridSearch')

# ---------------------------------------------------------
# Modelle laden (gecached)
# ---------------------------------------------------------
knn_model, _, _, _, _, _ = get_knn_model(X_train, y_train)
rf_model, _, _, _, _, _ = get_rf_model(X_train, y_train)

if model_choice == 'RandomForest mit RFECV':
    rfecv_model, _, _, _, _, _, rfecv, X_train_selected = get_rfecv_model(X_train, y_train)
    X_test_selected = rfecv.transform(X_test)


# ---------------------------------------------------------
# Tabs: Estimator / Hyperparameters / Feature Selection (für RFECV)
# ---------------------------------------------------------
if model_choice == 'RandomForest mit RFECV':
    tab1, tab2, tab3 = st.tabs(['Estimator', 'Hyperparameters', 'Feature Selection'])
    
    with tab1:
        st.write(best_model)
    
    with tab2:
        st.json(best_model.get_params())
    
    with tab3:
        st.subheader('RFECV Feature Selection Ergebnisse')
        st.metric('Optimale Anzahl Features', rfecv.n_features_)
        st.metric('Originale Features', X_train.shape[1])
        st.metric('Reduktion', f'{(1 - rfecv.n_features_/X_train.shape[1])*100:.1f}%')
        
        selected_features = X_train.columns[rfecv.support_].tolist()
        removed_features = X_train.columns[~rfecv.support_].tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Behaltene Features')
            st.write(f'({len(selected_features)} Features)')
            for f in selected_features[:15]:
                st.write(f'- {f}')
            if len(selected_features) > 15:
                st.write(f'... und {len(selected_features)-15} weitere')
        
        with col2:
            st.subheader('Entfernte Features')
            st.write(f'({len(removed_features)} Features)')
            for f in removed_features[:15]:
                st.write(f'- {f}')
            if len(removed_features) > 15:
                st.write(f'... und {len(removed_features)-15} weitere')
else:
    tab1, tab2 = st.tabs(['Estimator', 'Hyperparameters'])
    
    with tab1:
        st.write(best_model)
    
    with tab2:
        if model_choice == 'KNN':
            st.json(best_model.named_steps['knn'].get_params())
        else:
            st.json(best_model.get_params())


# ---------------------------------------------------------
# CV Score
# ---------------------------------------------------------
st.write('Best CV score:', score)


# ---------------------------------------------------------
# Learning Curve Plot
# ---------------------------------------------------------
st.header('Learning Curve')

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_std = test_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_sizes, train_mean, 'o-', label='Training Score')
ax.plot(train_sizes, test_mean, 'o-', label='Validation Score')

ax.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.2)
ax.fill_between(train_sizes, test_mean-test_std, test_mean+test_std, alpha=0.2)

ax.set_title(f'Learning Curve - {model_choice}')
ax.set_xlabel('Training Size')
ax.set_ylabel('Accuracy')
ax.legend()
ax.grid(True)

st.pyplot(fig)


# ---------------------------------------------------------
# Shapley für RF
# ---------------------------------------------------------
st.header('Shapley Values beim RandomForest')
if model_choice == 'RandomForest' or model_choice == 'RandomForest mit RFECV':
    st.image('images/beeswarm.png')


# ---------------------------------------------------------
# Feature Importances für RF
# ---------------------------------------------------------
if model_choice == 'RandomForest' or model_choice == 'RandomForest mit RFECV':
    st.header('Feature Importances')
    st.image('images/feature_importance_rf.png')


# ---------------------------------------------------------
# PCA für KNN
# ---------------------------------------------------------
if model_choice == 'KNN':
    st.header('KNN - Datenvisualisierung mit PCA')
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train)
    
    fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=y_train,
                     title=f'PCA (erklaert {pca.explained_variance_ratio_.sum():.1%} der Varianz)')
    st.plotly_chart(fig)
    
    st.caption(f'k={best_model.named_steps['knn'].n_neighbors if hasattr(best_model, 'named_steps') else best_model.n_neighbors} Nachbarn')


# ---------------------------------------------------------
# Modelle vergleichen
# ---------------------------------------------------------
st.header('McNemar-Test: Modellvergleich')

compare_choice = st.selectbox('Vergleiche', ['KNN vs RandomForest', 'KNN vs RFECV-RandomForest'])

if compare_choice == 'KNN vs RandomForest':
    result = compare_models_mcnemar(knn_model, rf_model, X_test, y_test)
    st.write(result['interpretation'])
    st.subheader('Kontingenztabelle')
    contingency_df = pd.DataFrame(
        result['contingency'],
        index=['RF richtig', 'RF falsch'],
        columns=['KNN richtig', 'KNN falsch']
    )
    st.dataframe(contingency_df)
    
    st.metric('KNN Accuracy', f"{result['acc_A']:.4f}")
    st.metric('RandomForest Accuracy', f"{result['acc_B']:.4f}")
    st.metric('p-Wert', f"{result['pvalue']:.4f}")

else:
    if model_choice == 'RandomForest mit RFECV':
        result = compare_models_mcnemar_with_rfecv(knn_model, rfecv_model, X_test, X_test_selected, y_test)
        st.write(result['interpretation'])
        st.subheader('Kontingenztabelle')
        contingency_df = pd.DataFrame(
            result['contingency'],
            index=['RFECV richtig', 'RFECV falsch'],
            columns=['KNN richtig', 'KNN falsch']
        )
        st.dataframe(contingency_df)
        
        st.metric('KNN Accuracy', f"{result['acc_knn']:.4f}")
        st.metric('RFECV-RandomForest Accuracy', f"{result['acc_rfecv']:.4f}")
        st.metric('p-Wert', f"{result['pvalue']:.4f}")
    else:
        st.warning('Bitte waehle zuerst RandomForest mit RFECV als Modell aus, um diesen Vergleich zu verwenden.')
        result = compare_models_mcnemar(knn_model, rf_model, X_test, y_test)
        st.write(result['interpretation'])
        st.subheader('Kontingenztabelle (KNN vs RF)')
        contingency_df = pd.DataFrame(
            result['contingency'],
            index=['RF richtig', 'RF falsch'],
            columns=['KNN richtig', 'KNN falsch']
        )
        st.dataframe(contingency_df)