
import pandas as pd
import numpy as np

def make_df(housing_file, pop_file):
    """
    The data in the two files are read and merged into a single DataFrame using nta2010 and NTA Code as the keys.
    If the total is null or Year differs from 2010, that row is dropped. The columns the_geom, boro, and nta2010 are dropped, and the resulting DataFrame is returned.
    """
    house = pd.read_csv (housing_file)  
    pop = pd.read_csv (pop_file)     #read in the file
    df = pd.merge (house, pop, how='left', left_on="nta2010", right_on= "NTA Code")
    df = df[df['Year']== 2010]                      #Only rows containing data for 2010
    df = df[df['total'].notnull()]                  #using not null function to get rid of null total
    droppingcol = ['the_geom', 'boro', 'nta2010']   #these columns are dropped
    df= df.drop(droppingcol, axis=1) 
    return df 
def compute_lin_reg(x,y): 
    """
    The function returns two numeric values: theta_0,theta_1 where
    theta_0 is the y-intercept of the best fitting line for x and y.
    theta_1 is the slope of the best fitting line for x and y.
    """
    r = x.corr(y)
    theta_1 = (r*(y.std())/(x.std())) 
    theta_0 = ((y.mean()) - theta_1*(x.mean()))
    return theta_0, theta_1

def compute_boro_lr(df,xcol,ycol,boro=["All"]):
    #If boro is ['All'], this function behaves identically to compute_lin_reg(df[xcol],df[ycol])
    if boro ==["All"]: 
        return compute_lin_reg(df[xcol], df[ycol]) 
    #Otherwise, the DataFrame is restricted to rows with Borough in boro and the restricted DataFrame is used, 
    else:
        df = df.loc[df['Borough'].isin(boro)]
        return compute_lin_reg(df[xcol], df[ycol]) 

def MSE_loss(y_actual,y_estimate): #the mean square error loss function 
    return ((y_estimate - y_actual)** 2).mean()
  

def RMSE(y_actual,y_estimate): #square root of the mean square error loss function
    return np.sqrt(((y_estimate - y_actual)** 2).mean())
    

def compute_error(y_actual,y_estimate,loss_fnc=MSE_loss): #use the loss function with the provided parameters 
    return loss_fnc(y_actual,y_estimate)