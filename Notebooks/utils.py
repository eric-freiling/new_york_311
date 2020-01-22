import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder# StandardScaler, 
from datetime import datetime
import os
 #import xgboost as xgb
# from sklearn.cross_validation import cross_val_score
from sklearn import metrics
import psycopg2

from fastai import *
from fastai.tabular import *

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

save_path = Path('../Saved_Files')


def query_db(query):
    '''
    Runs sql query on 311 database and returns a pandas DataFrame.
    Redshift is a data warehouse based on PostgreSQL, so syntax is mostly the
    same
    '''
    host = 'interview-ds.ckgnwnm6pw4o.us-east-1.redshift.amazonaws.com'
    port = 5439
    db = 'interview'
    username = 'dsguest'
    password = 'nX9EFYUZ5Yu#0q'
    conn = psycopg2.connect(host=host, port=port, dbname=db, user=username,
    password=password)
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()

    rows = pd.DataFrame(rows)
    return rows

    
def get_col_classes():
    cont_cols = [
        'x_coordinate_state_plane',
        'y_coordinate_state_plane',
        'latitude',
        'longitude',
        'time_to_close',
        'due_len',
        'time_over'
    ]
    cat_cols = [
        'agency',
        'borough',
        'location_type',
        'incident_zip',
        'street_name',
        'cross_street_1',
        'cross_street_2',
        'intersection_street_1',
        'intersection_street_2',
        'address_type',
        'city',
        'landmark',
        'facility_type',
        'status',
        'community_board',
        'open_data_channel_type',
        'park_facility_name',
        'park_borough',
        'vehicle_type',
        'taxi_company_borough',
        'taxi_pick_up_location',
        'bridge_highway_name',
        'bridge_highway_direction',
        'road_ramp',
        'bridge_highway_segment',
    ]
    date_cols = [
        'created_date',
        'closed_date',
        'due_date',
    ]
    dep_var = ['complaint_type']
    
    drop_cols = [
        'unique_key',                     # If interested in leakage, investigate this column
        'agency_name',                    # Redundant to agancy
        'descriptor',                     # This gives away the complaint type, too easy
        'incident_address',               # Didnt want street numbers, too easy
        'bbl',                            # We already have enough goelocaions
        'location',                       # Redundant to Lat and Lng
        'resolution_action_updated_date', # caused errors
        # Maybe Use if there is time
        'resolution_description',         # Probably very useful but not enough time to use
        
    ]
    
    return cont_cols, cat_cols, date_cols, drop_cols, dep_var


def filter_rows(df):
    labels = np.load(save_path / 'labels.npy')
    df = df[df['complaint_type'].isin(labels)]
    df = df.dropna(how='all')
    return df 


def drop_useless_cols(df, drop_cols):
    
    return df.drop(columns=drop_cols)


def create_date_lengths(df, cont_cols):
    df['time_to_close'] = (df['closed_date']-df['created_date']).astype('timedelta64[h]')
    df['due_len'] = (df['due_date']-df['created_date']).astype('timedelta64[h]')
    df['time_over'] = (df['due_date']-df['closed_date']).astype('timedelta64[h]')
    cont_cols += ['time_to_close', 'due_len', 'time_over']
    
    return df, cont_cols


def process_date_cols(df, cont_cols, cat_cols, date_cols, dep_var):
    # Convert all date cols to datetimes
    for dc in date_cols:
        df[dc] = pd.to_datetime(df[dc])

    # add columns for time elapsed between dates
    df, cont_cols = create_date_lengths(df, cont_cols)
    
    # Add date boolean features, day of week, end of year, etc
    for d in date_cols:
        add_datepart(df, d, drop=True)
    
    # Keep track of created categorical columns
    cat_cols += list(set(df.columns) - set(cont_cols) - set(cat_cols) - set(dep_var))

    return df, cont_cols, cat_cols

    
def fill_missing(df, cont_cols, cat_cols):
    # We fill in missing values with the median and add a new column to 
    # indicate if the data was missing. 
    
    nan_cols = list(set(df.columns[df.isna().any()]) & set(cont_cols))
    for nc in nan_cols:
        nan_loc = np.where(df[nc].isna())[0]
        good_loc = np.where(df[nc].notna())[0]
        med = np.median(df[nc].values[good_loc])
        x = df[nc].values
        
        # Fill missing values with median
        x[nan_loc] = med
        indicator = np.zeros(len(x))
        indicator[nan_loc] = 1

        # Define missing column
        cat_cols.append(nc + '_missing')
        df[nc + '_missing'] = indicator
    
    return df, cont_cols, cat_cols
       
def process_cont_cols(df, cont_cols, cat_cols):
    # Take care of missing data
    df, cont_cols, cat_cols = fill_missing(df, cont_cols, cat_cols)
    
    # Normalize column
    for cc in cont_cols:
        x = df[cc].values
        stand_dev = np.std(x)
        assert(stand_dev > 0)
        df[cc] = (x - np.mean(x)) / stand_dev

    return df, cont_cols, cat_cols
        
def combine_dfs(a, b):
    a.reset_index(drop=True, inplace=True)
    b.reset_index(drop=True, inplace=True)

    return pd.concat([a, b], axis=1)

def process_cat_cols(df, cat_cols, ohe_limit=20):
    # Categorical variables with number of unique values less than ohe limit get 
    # a One hot encoding, otherwise ordinal 
    
    # Fill missing data with string, precaution so everything is treated the same
    # and different algorithms might not handle it
    nan_cols = list(set(df.columns[df.isna().any()]) & set(cat_cols))
    for nc in nan_cols:
        # df.loc[:, nc] = df[nc].fillna('missing')
        df[nc] = df[nc].fillna('missing')

    # Go through categorical columns
    label_encoders = {}
    new_col_names = []
    new_cols = []
    new_drop_cols = []
    for cc in cat_cols:
        # make sure everything is a string
        x = [str(v) for v in df[cc].values]
        if df[cc].nunique() > ohe_limit:
            # Ordinal Encoding
            le = LabelEncoder()
            label_encoders[cc] = le
            df[cc] = le.fit_transform(x)
            
        else:
            # One Hot Encoding
            dummies = pd.get_dummies(x)
            if type(df[cc].values[0]) == np.bool_:
                dummies.columns = [cc + '=false', cc + '=true']
            else:
                dummies.columns = [cc + '=' + n for n in dummies.columns]
            # df = combine_dfs(df, dummies)
            new_col_names += list(dummies.columns)
            # neccessary for concatenating dfs
            dummies.reset_index(drop=True, inplace=True)
            new_cols.append(dummies)
            new_drop_cols.append(cc)
            
    df = df.drop(columns=new_drop_cols)
    df.reset_index(drop=True, inplace=True)
    df = pd.concat([df] + new_cols, axis=1)
    cat_cols += new_col_names
    
    return df, cat_cols, label_encoders
    

def split_target(df, dep_var, label_encoders):
    target = [str(v) for v in df[dep_var].values]
    le = LabelEncoder()
    label_encoders[dep_var] = le
    y = le.fit_transform(target)
    x = df.drop(columns=dep_var)
    
    return x, y, label_encoders
    
def transform_df(df):
    cont_cols, cat_cols, date_cols, drop_cols, dep_var = get_col_classes()
    print('Dropping useless columns...')
    df = drop_useless_cols(df, drop_cols)   
    print('Filter Rows...')
    df = filter_rows(df)
    print('Process Date Columns...')
    df, cont_cols, cat_cols = process_date_cols(df, cont_cols, cat_cols, date_cols, dep_var)
    print('Process Cont Columns...')
    df, cont_cols, cat_cols = process_cont_cols(df, cont_cols, cat_cols)
    print('Process Cat Columns...')
    df, cat_cols, label_encoders = process_cat_cols(df, cat_cols)
    
    return df, cont_cols, cat_cols, label_encoders
    
    
if __name__ == "__main__":
    # file_path = "../Interview_Project/Medical Appointments.csv"
    # df = pd.read_csv(file_path, encoding='latin-1')
    # nn(df)
    # rand_f(df)
    rf_hyperparam()