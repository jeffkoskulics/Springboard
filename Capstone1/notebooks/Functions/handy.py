import pandas as pd

def ModelMatrix(pipeline, row):
    row_names = row.columns.tolist()
    BigMountain_data = row.iloc[0].tolist()
    col_names = ['value','BigMountain']
    d = {'value':row_names, 'BigMountain':BigMountain_data}

    print(len(pipeline.named_steps['estimator'].coef_))

    model_matrix = pd.DataFrame()
    model_matrix['feature'] = row_names
    model_matrix['BgMtn'] = BigMountain_data
    model_matrix['mean'] = pipeline.named_steps['scale'].mean_
    model_matrix['BgMtn-mean'] = model_matrix['BgMtn'] - model_matrix['mean']
    model_matrix['stdev'] = pow(pipeline.named_steps['scale'].var_,0.5)
    model_matrix['BgMtnFeat'] = model_matrix['BgMtn-mean'] / model_matrix['stdev']
    model_matrix['$/feat'] = pipeline.named_steps['estimator'].coef_[0,:]
    model_matrix['BgMtnValue'] = model_matrix['BgMtnFeat'] * model_matrix['$/feat']
    print(model_matrix[['feature','BgMtnFeat','$/feat','BgMtnValue']].sort_values('$/feat', ascending=False).round(2))
    return model_matrix
