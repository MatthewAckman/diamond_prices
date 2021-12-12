'''
modular application of fitted interpolation for dealing with missing values
'''

def fill_nans(dataframe):
    '''
    takes a pandas dataframe and fills nans according to criteria
    '''
    import statsmodels.api as sm

    
