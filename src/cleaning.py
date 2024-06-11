# Caster les types des variables du df dans le type approprié
def convertType(df, intToFloat, floatToInt):
    for col in intToFloat:
        df[col] = df[col].astype('float64')
    for col in floatToInt:
        df[col] = df[col].astype('int64')
    return df

# Fonction pour filtrer les valeurs aberrantes
def filter_outliers(df, col_limits):
    for col, limit in col_limits.items():
        df = df[df[col] < limit]
    return df