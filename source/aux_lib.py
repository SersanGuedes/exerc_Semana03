import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

#a = pd.DataFrame([7,5,3,6,2,4])

#path_file = r"C:\Users\sersan.guedes\Desktop\exerc_Semana03\data\portland-oregon.csv"
#df1 = pd.read_csv(path_file, index_col='Month', parse_dates=['Month'])


def FNC_insert_N_Features( df, N ):
    v_cols = df.columns
    df = df.rename( columns = { v_cols[0]: "Feature1" } )   

    for jj in list( range( 1, N+1 ) ):
        df.insert( loc = jj, column = "Feature" + str(jj+1), value = df.Feature1.shift(-jj) )   

    v_cols = df.columns
    df = df.rename( columns = { v_cols[-1]: "Target" } )   

    df = df.dropna()

    return df

def FNC_Xy_portland( N ):
    path_file = r"C:\Users\sersan.guedes\Desktop\exerc_Semana03\data\portland-oregon.csv"
    df1 = pd.read_csv(path_file, index_col='Month', parse_dates=['Month'])
    
    df2 = FNC_insert_N_Features( df1, N )

    #X = df2.Count.values[0:-1].reshape(-1,1)
    #y = df2.Target[0:-1]

    X = df2.drop( columns = 'Target' )
    y = df2.loc[ :, 'Target' ]

    return X, y, df1, df2

#b = FNC_insert_N_Features( df1, 3 )
N = 4
X, y, df1, df2 = FNC_Xy_portland( N )

reg = LinearRegression()

reg.fit(X, y)
y_pred = reg.predict(X)

mean_squared_error( y_pred, y )

x_proxMes = df1.iloc[-N:,0].values
x_proxMes = np.array( x_proxMes ).reshape( 1, -1 )

y_proxMes = reg.predict( x_proxMes )
a = reg.predict( np.array([1425, 1419, 1432, 1394]).reshape(1, -1) )

plt.figure(101)
plt.plot( y, label='y' )
plt.plot( y_pred, label='y_pred' )
plt.legend()