import feather
import pandas as pd


test22 = pd.read_csv("test22.csv")
test22 = test22.dropna()
test22["sasdate"] = test22["sasdate"].apply(lambda x: x[-4:] + '/' + x[:-5])
test22["sasdate"] = pd.to_datetime(test22["sasdate"])
test22 = test22.set_index(["sasdate"])
feather.write_dataframe(test22, 'BRinf.feather')
