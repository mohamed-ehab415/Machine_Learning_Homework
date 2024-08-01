import numpy as np

from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold ,GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
def import_data():

  df=pd.read_csv(r"C:\Users\lap shop\Desktop\machin\5 data concept\10 Homework 2 - Regressors Project - 10 tasks\data.csv")
  df.fillna(df.median())
  data=df.to_numpy()
  X=data[:, :-1]
  t=data[:,-1].reshape(-1, 1)
  return  X ,t

def monomials_poly_features(X,degree):
    exambles=[]
    for examble in X:
        exambles_feat=[]
        for features in examble:
            curr=1
            each_feat=[]
            for deg in range(degree):
                curr*=features
                each_feat.append(curr)
            exambles_feat.extend(each_feat)
        exambles.append(np.array(exambles_feat))
    return np.vstack(exambles)








def Transformation(X):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def Regularized_polynomial_regression(X,t):
    X_poly=monomials_poly_features(X,2)

    cv=KFold(n_splits=4,shuffle=True,random_state=17)
    alphas = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    param_grid = {'model__alpha': alphas}  # Updated param_grid to include model__alpha
    pipeline = Pipeline(steps=[("scaler", MinMaxScaler()), ('model', Ridge())])
    grid=GridSearchCV(pipeline,param_grid,cv=cv, scoring='neg_mean_squared_error')
    grid.fit(X_poly,t)
    rmses = np.sqrt(-grid.cv_results_['mean_test_score'])
    for (alpha, rmse) in zip(alphas, rmses):
        print(f'alpha = {alpha} - rmse = {rmse}', end='')
        if alpha == grid.best_params_['model__alpha']:  # Updated to model__alpha
            print('\t\t**BEST PARAM**')
        else:
            print()

    plt.title(f'log10(Alphas) for degree {2} vs cross validation RMSE')  # Corrected typo in the title
    plt.xlabel('log10(alpha)')
    plt.ylabel('RMSE')

    # visualizing with alpha is hard. Use log10
    alphas = np.log10(alphas)
    plt.xticks(alphas)
    plt.plot(alphas, rmses, label='train error')

    plt.grid()
    plt.show()










if __name__ == "__main__":
  X,t=import_data()

  Regularized_polynomial_regression(X,t)
