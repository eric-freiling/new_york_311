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

from fastai import *
from fastai.tabular import *

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

save_path = Path('../Saved_Models')


# file_path = "../Interview_Project/Medical Appointments.csv"
# df = pd.read_csv(file_path)
# keys = list(df.keys())
# values = df.values
# answers = values[:, -1]
# num_rows = len(answers)
# y = np.zeros(num_rows)
# y[answers == "Yes"] = 1
#
# no_show_percent = np.sum(y) / len(y)
#
# x = df[[
#     'Gender',
#     'ScheduledDay',
#     'AppointmentDay',
#     'Age',
#     'Neighbourhood',
#     'Scholarship',
#     'Hipertension',
#     'Diabetes',
#     'Alcoholism',
#     'Handcap',
#     'SMS_received'
# ]]
# y=df['No-show']  # Labels
# # Split dataset into training set and test set
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


def normalize_data(df, ohe=True):
    df["AppointmentDay_DT"] = pd.to_datetime(df['AppointmentDay'])
    df["ScheduledDay_DT"] = pd.to_datetime(df['ScheduledDay'])
    df["ScheduledDayOfWeek"] = df['ScheduledDay_DT'].dt.dayofweek
    df["AppointmentDayOfWeek"] = df['AppointmentDay_DT'].dt.dayofweek
    y_str = df["No-show"].values
    y = np.zeros(len(y_str))
    y[y_str == "Yes"] = 1
    cols = [
        "Age",
        "Scholarship",
        "Hipertension",
        "Alcoholism",
        "Diabetes",
        "SMS_received",
        "ScheduledDayOfWeek",
        "AppointmentDayOfWeek"
    ]
    if ohe:
        df2 = pd.DataFrame(df[cols].values, columns=cols)
        # df2["Age"] = (df["Age"] - df["Age"].mean()) / df["Age"].std()
        le_gender = LabelEncoder()
        df2["Gender"] = le_gender.fit_transform(df["Gender"].values)
        dummies_age =  pd.get_dummies(df["Age"])
        dummies_hndcp = pd.get_dummies(df["Handcap"])
        dummies_nbhd = pd.get_dummies(df["Neighbourhood"])

        return pd.concat([df2, dummies_hndcp, dummies_nbhd, dummies_age], axis=1), y
    else:
        cols = cols + ["Neighbourhood", "Handcap"]
        df2 = pd.DataFrame(df[cols].values, columns=cols)
        df2["Age"] = (df["Age"] - df["Age"].mean()) / df["Age"].std()
        le = LabelEncoder()
        le_gender = LabelEncoder()
        df2["Neighbourhood"] = le.fit_transform(df2["Neighbourhood"].values)
        df2["Gender"] = le_gender.fit_transform(df["Gender"].values)

        return df2, y


def acc(y, y_hat):
    return round(np.sum(y == y_hat)*100 / len(y), 2)


def random_forest(x_train, x_test, y_train, y_test, cols, depth=4, n_est=100):
    rf = RandomForestClassifier(max_depth=depth, n_estimators=n_est, class_weight="balanced_subsample")
    rf.fit(x_train, y_train)
    y_hat_train = rf.predict(x_train)
    y_hat_test = rf.predict(x_test)

    train_acc = acc(y_hat_train, y_train)
    test_acc = acc(y_hat_test, y_test)
    print("Training Accuracy: {}".format(train_acc))
    print("Testing Accuracy: {}".format(test_acc))
    sort_ind = np.argsort(rf.feature_importances_)
    cols = np.array(cols)
    print("Feature Importance: ", list(zip(cols[sort_ind], rf.feature_importances_[sort_ind])))
    return rf


def define_seq_model(input_shape, hidden_layers, output_size,
                     learning_rate=0.01, drop_rate=0.5, hidden_act='relu', final_act='sigmoid'):
    # first input model
    model = Sequential()
    model.add(Dense(hidden_layers[0], activation=hidden_act, input_dim=input_shape))
    model.add(Dropout(drop_rate))
    for i in range(1, len(hidden_layers)):
        model.add(Dense(hidden_layers[i], activation=hidden_act))
        model.add(Dropout(drop_rate))
    model.add(Dense(output_size, activation=final_act))
    opt = rmsprop(lr=learning_rate)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


def get_batch(x, y, batch_size):
    indices_0 = np.where(y == 0)[0]
    indices_1 = np.where(y == 1)[0]

    num_0 = len(indices_0)
    num_1 = len(indices_1)

    indices_0_ind = np.random.randint(0, num_0, size=batch_size // 2)
    indices_1_ind = np.random.randint(0, num_1, size=batch_size // 2)

    indices = np.concatenate([indices_0[indices_0_ind], indices_1[indices_1_ind]])
    np.random.shuffle(indices)

    return x[indices, :], y[indices]


def save_model_weights(model, weight_path):
    # Usually I just use model.save_model() but there seems to be a bug when using a windows machine
    W = model.get_weights()
    np.save(weight_path, W)


def get_model(weight_path, input_shape, hidden_layers, output_size):
    model = define_seq_model(input_shape, hidden_layers, output_size)
    W = np.load(weight_path + ".npy")
    model.set_weights(W)
    return model


def train_nn(x_train, y_train, x_test, y_test):

    model_name = 'nn_4'
    batch_size = 10000
    epochs = 5000
    input_size = x_train.shape[1]
    hidden_layers = [100, 100, 100, 50]
    output_size = 1
    learning_rate = 0.01
    print_iter = 100
    save_iter = 100

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_path = os.path.join(save_dir, model_name)

    # Create folder to save models
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Check to see if model already exists
    # If exists load the model and pick up training where we left off
    # If doesn't exist, then create it
    if os.path.exists(model_path + ".npy"):
        print("Loading Saved Model...\n")
        model = get_model(model_path, input_size, hidden_layers, output_size)
        # model = load_model(model_path)
    else:
        print("Creating New Model...\n")
        model = define_seq_model(input_size, hidden_layers, output_size, learning_rate=learning_rate)

   # Run Epochs
    print("Starting to Train...\n")

    for i in range(epochs):
        # Load training batch
        x_batch, y_batch = get_batch(x_train, y_train, batch_size)

        # Train on batch
        model.train_on_batch(x_batch, y_batch)

        # Print updates
        if i % print_iter == 0:
            print("Epoch {}/{}".format(i, epochs))
            score = model.evaluate(x_batch, y_batch, verbose=0)
            loss = round(score[0], 2)
            acc = round(score[1], 2)
            print('Training ---- loss: {} accuracy: {}'.format(loss, acc))


            x_batch, y_batch = get_batch(x_test, y_test, batch_size)
            score = model.evaluate(x_batch, y_batch, verbose=0)
            loss = round(score[0], 2)
            acc = round(score[1], 2)
            print('Validation -- loss: {} accuracy: {}\n'.format(loss, acc))

        if i % save_iter == 0:
            save_model_weights(model, model_path)
            # model.save(model_path)

    save_model_weights(model, model_path)
    # model.save(model_path)
    print('\nSaved trained model at %s ' % model_path)



def nn(df):
    x, y = normalize_data(df)
    x = x.values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    train_nn(x_train, y_train, x_test, y_test)


def rand_f(df):
    x, y = normalize_data(df, ohe=False)
    cols = list(x.keys())
    x = x.values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    random_forest(x_train, x_test, y_train, y_test, cols)


def cross_val():

    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_curve, auc
    cv = StratifiedKFold(n_splits=10)

    depth = [4, 6, 8, 10, 12, 14]
    n_trees = np.linspace(20, 200, 10).astype(int)
    file_path = "../Interview_Project/Medical Appointments.csv"
    df = pd.read_csv(file_path, encoding='latin-1')
    x_lab, y = normalize_data(df, ohe=False)
    x_train, x_test, y_train, y_test = train_test_split(x_lab.values, y, test_size=0.2)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        rf = RandomForestClassifier(max_depth=5, n_estimators=50, class_weight="balanced_subsample")
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
def evaluate_prediction(y, y_hat):
    loc_0 = np.where(y == 0)[0]
    loc_1 = np.where(y == 1)[0]
    acc = round(np.sum(y == y_hat) / len(y_hat) * 100, 2)
    acc_0 = round(np.sum(y[loc_0] == y_hat[loc_0]) / len(y_hat[loc_0]) * 100, 2)
    acc_1 = round(np.sum(y[loc_1] == y_hat[loc_1]) / len(y_hat[loc_1]) * 100, 2)
    f1_score = round(metrics.f1_score(y, y_hat), 2)
    print("Accuracy: {}".format(acc))
    print("Accuracy 0: {}".format(acc_0))
    print("Accuracy 1: {}".format(acc_1))
    print("F1: {}".format(f1_score))


def rf_hyperparam():
    depth = [4, 8, 12, 16]
    n_trees = np.linspace(50, 200, 4).astype(int)
    file_path = "../Interview_Project/Medical Appointments.csv"
    df = pd.read_csv(file_path, encoding='latin-1')
    x_lab, y = normalize_data(df, ohe=False)
    x_train, x_test, y_train, y_test = train_test_split(x_lab.values, y, test_size=0.2)
    tree_scores = []
    for t in n_trees:
        rf = RandomForestClassifier(max_depth=depth[0], n_estimators=t, class_weight="balanced_subsample")
        cross_val = cross_val_score(rf, x_train, y_train, cv=10, scoring="f1")
        tree_scores.append(cross_val)
    tree_scores = np.array(tree_scores)
    np.save("tree_scores", tree_scores)
    tree_avg = np.mean(tree_scores, axis=1)
    idx = np.argmax(tree_avg)
    num_trees = n_trees[idx]
    print("Best num trees: ", num_trees)

    depth_scores = []
    for d in depth:
        rf = RandomForestClassifier(max_depth=d, n_estimators=50, class_weight="balanced_subsample")
        cross_val = cross_val_score(rf, x_train, y_train, cv=10, scoring="f1")
        depth_scores.append(cross_val)
    depth_scores = np.array(depth_scores)
    np.save("depth_scores", depth_scores)
    depth_avg = np.mean(depth_scores, axis=1)
    idx = np.argmax(depth_avg)
    print("Best depth: ", depth[idx])


    
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
    return df[df['complaint_type'].isin(labels)]


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
    cat_cols += new_cols
    
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