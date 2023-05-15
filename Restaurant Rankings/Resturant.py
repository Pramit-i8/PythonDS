import pandas as pd

def make_insp_df(file_name):
    """
    should open the file 'file_name' as DataFrame,

    drop everything except for the columns give to us in the dirrection
    """
    df = pd.read_csv(file_name,low_memory=False) #read in the file
    df = df[['CAMIS', 'DBA', 'BORO', 'BUILDING', 'STREET', 'ZIPCODE', 'SCORE', 'GRADE', 'NTA']] #keeping these columns
    df = df[ df['SCORE'].notnull() ] #learned this from the hackerrank quiz
    return df 

def predict_grade(num_violations): 
    """
    The function should then return the letter grade that corresponds to the 
    number of violation points num_violations
    "A" grade: 0 to 13 points
    "B" grade: 14 to 27 points
    "C" grade: 28 or more points
    """
    #just assigning letter grades to each violation number now 
    #using if statements
    if  13 >= num_violations >= 0: 
        return "A" 
    if  27 >= num_violations >= 14: 
        return "B"
    #if  num_violations >= 28: //alternative way to do this 
    #    return "C"
    else:
        return "C"

def grade2num(grade):
    """
     returns the grade on a 4.0 scale for grade = 'A', 'B', or 'C' (i.e. 4.0, 3.0, or 2.0, respectively).
    """
    if grade == 'A':
        return 4.0
    if grade == 'B':
        return 3.0
    if grade == 'C':
        return 2.0 
    #If grade is None or some other value, return None.
    else: 
        return None

def make_nta_df(file_name):
    df = pd.read_csv(file_name,low_memory=False) #read in the file
    df = df[['NTACode', 'NTAName', ]] #keeping only these columns
    return df

def compute_ave_grade(df,col):
    """
    This function returns a DataFrame with two columns, the NTACode and the average of col for each NTA.
    """
    #df=df[col].mean()
    df= df.groupby('NTA')[col].mean(numeric_only=True)
    df= df.to_frame()
    return df

def neighborhood_grades(ave_df,nta_df):

    df= pd.merge(ave_df, nta_df, how='left', left_on='NTA',right_on='NTACode')  
 #The columns NTA and NTACode should be dropped before returning the DataFrame.   
    droppingcol = ['NTA', 'NTACode']
    df= df.drop(droppingcol, axis=1) 
    return df