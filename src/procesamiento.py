from sklearn.preprocessing import LabelEncoder
#funcion para codificar las columnas X es el dataframe y exclude es un archivo que especifica que columnas se deben excluir

def encoder(X,exclude=[]):

    for col in X.select_dtypes(include=['object']).columns:
        if col not in exclude:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
    return X