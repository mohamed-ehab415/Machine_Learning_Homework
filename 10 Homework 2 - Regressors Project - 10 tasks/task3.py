import numpy as np

from task1 import import_data
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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









def Transformation(X_train, X_val):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled

def regreesion_model_for_train_and_val():
    X_train,t_train,X_Val,t_val=import_data()
    X_train_poly =monomials_poly_features(X_train,3)
    X_Val_poly=monomials_poly_features(X_Val,3)
    X_train_scaled,X_val_scaled,=Transformation(X_train_poly,X_Val_poly)

    model=LinearRegression(fit_intercept=True)
    model.fit(X_train_scaled,t_train)
    t_prd=model.predict(X_train_scaled)
    error_train=mean_squared_error(t_prd,t_train,squared=False)
     #for val

    t_prd_val=model.predict(X_val_scaled)
    error_val=mean_squared_error(t_prd_val,t_val,squared=False)


     #optimal_wieght and intercept
    intercept=model.intercept_
    optimal_wieght=abs(model.coef_).mean()
    return intercept,optimal_wieght,error_train,error_val



if __name__ == "__main__":
  intercept,optimal_wieght,error_train,error_val=regreesion_model_for_train_and_val()
  print("intercept = {} and optimal wieght= {}".format(intercept,optimal_wieght))
  print("error for train={}".format(error_train))
  print("error for val={}".format(error_val))
