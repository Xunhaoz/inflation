import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, BaggingRegressor, RandomForestRegressor, \
    HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import RANSACRegressor, LinearRegression, HuberRegressor, SGDRegressor, \
     PassiveAggressiveRegressor, BayesianRidge, ARDRegression, TweedieRegressor, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def get_X_y(df: pd.DataFrame):
    df = df.values
    X, y = df[:, 2:], df[:, 1].reshape(-1, 1)
    return X, y


if __name__ == "__main__":
    print('model preparing...')
    models = [
        ExtraTreesRegressor(),
        BaggingRegressor(),
        RandomForestRegressor(),
        HistGradientBoostingRegressor(),
        GradientBoostingRegressor(),
        AdaBoostRegressor(),
        DecisionTreeRegressor(),
        ExtraTreeRegressor(),
        SVR(max_iter=10000),
        NuSVR(max_iter=10000),
        LinearSVR(max_iter=10000),
        KNeighborsRegressor(),
        RANSACRegressor(),
        LinearRegression(),
        HuberRegressor(),
        SGDRegressor(),
        PassiveAggressiveRegressor(),
        BayesianRidge(),
        ARDRegression(),
        TweedieRegressor(),
        ElasticNet(),
        MLPRegressor(max_iter=10000),
        KernelRidge(),
        PLSRegression(),
        GaussianProcessRegressor(),
        XGBRegressor(),
        LGBMRegressor(),
        CatBoostRegressor(),
    ]

    print('data preparing...')
    test22 = pd.read_csv('test22.csv').dropna()

    in_sample, out_sample = test22[:int(len(test22) * 0.7)], test22[int(len(test22) * 0.7):]
    in_sample_X, in_sample_y = get_X_y(in_sample)
    out_sample_X, out_sample_y = get_X_y(out_sample)

    res = []
    for model in models:
        model_name = model.__class__.__name__

        print(model_name)
        model.fit(in_sample_X, in_sample_y.ravel())
        mse = mean_squared_error(model.predict(out_sample_X), out_sample_y)
        mae = mean_absolute_error(model.predict(out_sample_X), out_sample_y)

        res.append({
            "model": model_name,
            "mse": mse,
            "mae": mae
        })

        pd.DataFrame(res).to_csv('result.csv')
