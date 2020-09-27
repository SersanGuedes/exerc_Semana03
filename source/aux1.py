import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


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


def FNC_runTotal( N ):

    X, y, df1, df2 = FNC_Xy_portland( N )

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    reg = LinearRegression()

    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    rmse1 = mean_squared_error( y_pred, y_test, squared=False )

    x_proxMes = df1.iloc[-N:,0].values
    x_proxMes = np.array( x_proxMes ).reshape( 1, -1 )

    y_proxMes = reg.predict( x_proxMes )
    #a = reg.predict( np.array([1425, 1419, 1432, 1394]).reshape(1, -1) )

    print("RMSE: ", rmse1.round(2))

    #print("Previsão de nº de pessoas em 1969-06 (gab. é 1327): ", a)
    print("Previsão de nº de pessoas em 1969-07: ", y_proxMes)

    plt.figure(101)
    plt.plot( y_test.index, y_test, '*', label='y_test' )
    plt.plot( y_test.index, y_pred, '*', label='y_pred' )
    plt.legend()
    plt.show()

    print("FIM!")

    return y_test, y_pred


#def FNC_mnist():
