import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
def import_data():
  df=pd.read_csv(r"C:\Users\lap shop\Desktop\machin\5 data concept\10 Homework 2 - Regressors Project - 10 tasks\data.csv")
  df.fillna(df.median())
  data=df.to_numpy()
  X=data[:, :-1]
  t=data[:,-1].reshape(-1, 1)
  X_train=X[:100,:]
  t_train=t[:100,:]
  X_val=X[100:,:]
  t_val=t[100:,:]
  return X_train,t_train,X_val,t_val

def Transformation():
    X_train, t_train, X_val, t_val = import_data()
    scal = MinMaxScaler()
    X_train = scal.fit_transform(X_train)
    X_val = scal.transform(X_val)  # Use transform instead of fit_transform for the validation set

    return X_train, t_train, X_val, t_val





def regreesion_model_for_train_and_val():
     X_train,t_train,X_val,t_val=Transformation()

     model=LinearRegression(fit_intercept=True)
     model.fit(X_train,t_train)
     t_prd=model.predict(X_train)
     error_train=mean_squared_error(t_prd,t_train,squared=False)
     #for val

     t_prd_val=model.predict(X_val)
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

