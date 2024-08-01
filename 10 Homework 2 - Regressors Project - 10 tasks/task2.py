from task1 import import_data
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def poly(X_train, X_val, degree):
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train)
    X_val_poly = poly_features.transform(X_val)
    return X_train_poly, X_val_poly

def Transformation(X_train, X_val):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled

def regression_model_for_train_and_val(X_train, t_train, X_val, t_val):
    model = LinearRegression()
    model.fit(X_train, t_train)

    t_pred_train = model.predict(X_train)
    error_train = mean_squared_error(t_pred_train, t_train, squared=False)

    t_pred_val = model.predict(X_val)
    error_val = mean_squared_error(t_pred_val, t_val, squared=False)

    return error_train, error_val

def visizlion():
    error_train = []
    error_val = []
    degree = [1, 2, 3, 4]
    X_train, t_train, X_val, t_val = import_data()

    for i in degree:
        X_train_p, X_val_p = poly(X_train, X_val, i)
        X_train_scaled, X_val_scaled = Transformation(X_train_p, X_val_p)
        error_t, error_v = regression_model_for_train_and_val(X_train_scaled, t_train, X_val_scaled, t_val)
        error_train.append(error_t)
        error_val.append(error_v)

    if len(degree) > 1:
        plt.title('Degree vs train/val errors')
        plt.xlabel('Degree')
        plt.ylabel('RMSE')

        plt.xticks(degree)
        plt.plot(degree, error_train, label='error_train')
        plt.plot(degree, error_val, label='val error')
        plt.legend(loc='best')

        plt.grid()
        plt.show()

    return error_train, error_val

if __name__ == "__main__":
    visizlion()
