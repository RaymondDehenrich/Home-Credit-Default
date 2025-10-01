import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.decomposition import PCA
from lightgbm import LGBMClassifier
import sqlite3
import re


conn = sqlite3.connect("home_credit_default.db")

def fillMissing(tables:pd.DataFrame,Target=False):
    df = tables.copy()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if Target:
        if 'TARGET' in cat_cols:
            cat_cols.pop(cat_cols.index('TARGET'))
        if 'TARGET' in num_cols:
            num_cols.pop(num_cols.index('TARGET'))
    df[cat_cols] = df[cat_cols].fillna("Missing")
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    return df

def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def get_applicant_data():
    app_query ="""SELECT * FROM application_train
UNION ALL
SELECT SK_ID_CURR, NULL as TARGET, NAME_CONTRACT_TYPE, CODE_GENDER, FLAG_OWN_CAR, FLAG_OWN_REALTY, CNT_CHILDREN, AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE, NAME_TYPE_SUITE, NAME_INCOME_TYPE, NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, REGION_POPULATION_RELATIVE, DAYS_BIRTH, DAYS_EMPLOYED, DAYS_REGISTRATION, DAYS_ID_PUBLISH, OWN_CAR_AGE, FLAG_MOBIL, FLAG_EMP_PHONE, FLAG_WORK_PHONE, FLAG_CONT_MOBILE, FLAG_PHONE, FLAG_EMAIL, OCCUPATION_TYPE, CNT_FAM_MEMBERS, REGION_RATING_CLIENT, REGION_RATING_CLIENT_W_CITY, WEEKDAY_APPR_PROCESS_START, HOUR_APPR_PROCESS_START, REG_REGION_NOT_LIVE_REGION, REG_REGION_NOT_WORK_REGION, LIVE_REGION_NOT_WORK_REGION, REG_CITY_NOT_LIVE_CITY, REG_CITY_NOT_WORK_CITY, LIVE_CITY_NOT_WORK_CITY, ORGANIZATION_TYPE, EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, APARTMENTS_AVG, BASEMENTAREA_AVG, YEARS_BEGINEXPLUATATION_AVG, YEARS_BUILD_AVG, COMMONAREA_AVG, ELEVATORS_AVG, ENTRANCES_AVG, FLOORSMAX_AVG, FLOORSMIN_AVG, LANDAREA_AVG, LIVINGAPARTMENTS_AVG, LIVINGAREA_AVG, NONLIVINGAPARTMENTS_AVG, NONLIVINGAREA_AVG, APARTMENTS_MODE, BASEMENTAREA_MODE, YEARS_BEGINEXPLUATATION_MODE, YEARS_BUILD_MODE, COMMONAREA_MODE, ELEVATORS_MODE, ENTRANCES_MODE, FLOORSMAX_MODE, FLOORSMIN_MODE, LANDAREA_MODE, LIVINGAPARTMENTS_MODE, LIVINGAREA_MODE, NONLIVINGAPARTMENTS_MODE, NONLIVINGAREA_MODE, APARTMENTS_MEDI, BASEMENTAREA_MEDI, YEARS_BEGINEXPLUATATION_MEDI, YEARS_BUILD_MEDI, COMMONAREA_MEDI, ELEVATORS_MEDI, ENTRANCES_MEDI, FLOORSMAX_MEDI, FLOORSMIN_MEDI, LANDAREA_MEDI, LIVINGAPARTMENTS_MEDI, LIVINGAREA_MEDI, NONLIVINGAPARTMENTS_MEDI, NONLIVINGAREA_MEDI, FONDKAPREMONT_MODE, HOUSETYPE_MODE, TOTALAREA_MODE, WALLSMATERIAL_MODE, EMERGENCYSTATE_MODE, OBS_30_CNT_SOCIAL_CIRCLE, DEF_30_CNT_SOCIAL_CIRCLE, OBS_60_CNT_SOCIAL_CIRCLE, DEF_60_CNT_SOCIAL_CIRCLE, DAYS_LAST_PHONE_CHANGE, FLAG_DOCUMENT_2, FLAG_DOCUMENT_3, FLAG_DOCUMENT_4, FLAG_DOCUMENT_5, FLAG_DOCUMENT_6, FLAG_DOCUMENT_7, FLAG_DOCUMENT_8, FLAG_DOCUMENT_9, FLAG_DOCUMENT_10, FLAG_DOCUMENT_11, FLAG_DOCUMENT_12, FLAG_DOCUMENT_13, FLAG_DOCUMENT_14, FLAG_DOCUMENT_15, FLAG_DOCUMENT_16, FLAG_DOCUMENT_17, FLAG_DOCUMENT_18, FLAG_DOCUMENT_19, FLAG_DOCUMENT_20, FLAG_DOCUMENT_21, AMT_REQ_CREDIT_BUREAU_HOUR, AMT_REQ_CREDIT_BUREAU_DAY, AMT_REQ_CREDIT_BUREAU_WEEK, AMT_REQ_CREDIT_BUREAU_MON, AMT_REQ_CREDIT_BUREAU_QRT, AMT_REQ_CREDIT_BUREAU_YEAR FROM application_test"""
    app_df = pd.read_sql(app_query, conn)
    app_df,_ = one_hot_encoder(app_df)
    return app_df


def get_bureau_data():
    #Main Bureau and bal data
    bureau_query="""SELECT * FROM bureau"""
    bureau_query="""SELECT * FROM bureau"""
    bureau_bal_query = """SELECT * FROM bureau_balance"""
    bureau_df,_ = one_hot_encoder(pd.read_sql(bureau_query,conn))
    bureau_bal_df= pd.read_sql(bureau_bal_query,conn)
    bureau_bal_df,bureau_bal_new_col = one_hot_encoder(bureau_bal_df,True)
    bureau_bal_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bureau_bal_new_col:
        bureau_bal_aggregations[col] = ['any','sum']
    bureau_bal_agg = bureau_bal_df.groupby('SK_ID_BUREAU').agg(bureau_bal_aggregations)
    bureau_bal_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bureau_bal_agg.columns.tolist()])
    joined_bureau = bureau_df.join(bureau_bal_agg,on='SK_ID_BUREAU',how='left',rsuffix="_BB")

    #Final Aggregate
    joined_bureau = joined_bureau.drop(columns=['SK_ID_BUREAU'])
    bureau_aggregations ={'MONTHS_BALANCE_MIN':['min'],'MONTHS_BALANCE_MAX':['max'],'MONTHS_BALANCE_SIZE':['size']}
    for col in joined_bureau.columns:
        if 'STATUS' in col:
            if 'ANY' in col:
                bureau_aggregations[col]=['any']
            elif 'SUM' in col:
                bureau_aggregations[col]=['sum']
        elif col in ['DAYS_CREDIT','DAYS_CREDIT_ENDDATE','AMT_CREDIT_SUM_LIMIT']:
            bureau_aggregations[col] = ['mean']
        elif col in ['CREDIT_DAY_OVERDUE','CNT_CREDIT_PROLONG']:
            bureau_aggregations[col] = ['sum']
        elif 'CREDIT_TYPE_' in col or 'CREDIT_CURRENCY_' in col or 'CREDIT_ACTIVE_' in col:
            bureau_aggregations[col] = ['any','sum']
        elif col in ['AMT_CREDIT_MAX_OVERDUE','AMT_CREDIT_SUM','AMT_CREDIT_SUM_DEBT','AMT_CREDIT_SUM_OVERDUE','AMT_ANNUITY']:
            bureau_aggregations[col] = ['sum','mean']
    final_bureau_agg_df = joined_bureau.groupby('SK_ID_CURR').agg(bureau_aggregations)
    final_bureau_agg_df.columns = pd.Index([e[0] + "_" + e[1].upper() for e in final_bureau_agg_df.columns.tolist()])

    return final_bureau_agg_df
    

def get_prev_application_data():
    #POS_CASH_Balance
    pos_cash_bal_query = """SELECT * FROM POS_CASH_balance"""
    pos_cash_bal_df = pd.read_sql(pos_cash_bal_query,conn).drop(columns=['SK_ID_CURR'])
    pcb_df,pcb_new_col = one_hot_encoder(pos_cash_bal_df,True)
    pcb_aggregations = {'MONTHS_BALANCE':['min','max','size'],'CNT_INSTALMENT':['min','max','mean'],'CNT_INSTALMENT_FUTURE':['min','max','mean'],'SK_DPD':['mean','sum'],'SK_DPD_DEF':['mean','sum']}
    for col in pcb_new_col:
        pcb_aggregations[col] = ['any','sum']
    pcb_agg_df = pcb_df.groupby(['SK_ID_PREV']).agg(pcb_aggregations)
    pcb_agg_df.columns = pd.Index([e[0] + "_" + e[1].upper() for e in pcb_agg_df.columns.tolist()])

    #INSTALLMENTS
    installments_query = """SELECT * FROM installments_payments"""
    installments_df = pd.read_sql(installments_query,conn).drop(columns=['SK_ID_CURR'])
    installments_aggregations = {'NUM_INSTALMENT_VERSION':['nunique'], 'NUM_INSTALMENT_NUMBER':['max','min'],
        'DAYS_INSTALMENT':['min', 'max','mean'], 'DAYS_ENTRY_PAYMENT':['min', 'max', 'mean'], 'AMT_INSTALMENT':['sum','mean'],
        'AMT_PAYMENT':['mean', 'sum']}
    installments_agg_df = installments_df.groupby(['SK_ID_PREV']).agg(installments_aggregations)
    installments_agg_df.columns = pd.Index([e[0] + "_" + e[1].upper() for e in installments_agg_df.columns.tolist()])

    #CREDIT CARD
    credit_card_query = """SELECT * FROM credit_card_balance"""
    credit_card_df = pd.read_sql(credit_card_query,conn).drop(columns=['SK_ID_CURR'])
    credit_card_df,cc_new_col = one_hot_encoder(credit_card_df,True)
    cc_aggregations ={}
    for col in credit_card_df.columns:
        if 'NAME_CONTRACT_STATUS' in col:
            cc_aggregations[col]=['any','sum']
        if col in ['MONTH_BALANCE']:
            cc_aggregations[col]=['min','max','size']
        elif col in ['CNT_INSTALMENT_MATURE_CUM','CNT_DRAWINGS_OTHER_CURRENT','CNT_DRAWINGS_ATM_CURRENT','CNT_DRAWINGS_POS_CURRENT','CNT_DRAWINGS_CURRENT']:
            cc_aggregations[col]=['sum']
        elif col != 'SK_ID_PREV': cc_aggregations[col]=['mean','sum']
    cc_agg_df = credit_card_df.groupby(['SK_ID_PREV']).agg(cc_aggregations)
    cc_agg_df.columns = pd.Index([e[0] + "_" + e[1].upper() for e in cc_agg_df.columns.tolist()])

    #JOINED
    prev_app_query = """SELECT * FROM previous_application"""
    prev_app_df = pd.read_sql(prev_app_query,conn)
    prev_app_df,prev_app_new_col = one_hot_encoder(prev_app_df,True)
    prev_app_df['WEIGHTED_RATE_DOWN'] = prev_app_df['RATE_DOWN_PAYMENT'] * prev_app_df['AMT_CREDIT']
    prev_app_df['WEIGHTED_RATE_INTEREST_PRIMARY'] = prev_app_df['RATE_INTEREST_PRIMARY'] * prev_app_df['AMT_CREDIT']
    prev_app_df['WEIGHTED_RATE_INTEREST_PRIVILEGED'] = prev_app_df['RATE_INTEREST_PRIVILEGED'] * prev_app_df['AMT_CREDIT']
    prev_app_df = prev_app_df.drop(columns=['RATE_DOWN_PAYMENT','RATE_INTEREST_PRIMARY','RATE_INTEREST_PRIVILEGED'])


    joined_prev_app_df = prev_app_df.join(pcb_agg_df,on=['SK_ID_PREV'],how='left',rsuffix='_PCB')
    joined_prev_app_df = joined_prev_app_df.join(installments_agg_df,on=['SK_ID_PREV'],how='left',rsuffix='_INST')
    joined_prev_app_df = joined_prev_app_df.join(cc_agg_df,on=['SK_ID_PREV'],how='left',rsuffix='_CC')

    #Full Aggregrate
    joined_prev_app_df = joined_prev_app_df.drop(columns=['SK_ID_PREV'])
    final_agg ={}
    for col in joined_prev_app_df.columns:
        if joined_prev_app_df[col].dtypes=='bool':
            final_agg[col]=['any']
        elif 'SUM' in col:
            final_agg[col]=['sum']
        elif 'MEAN' in col or col in ['DAYS_DECISION','DAYS_FIRST_DRAWING','DAYS_FIRST_DUE','DAYS_LAST_DUE_1ST_VERSION','DAYS_LAST_DUE','DAYS_TERMINATION']:
            final_agg[col]=['mean']
        elif col in ['NFLAG_INSURED_ON_APPROVAL']:
            final_agg[col]=['mean','any','sum']
        elif col in ['SELLERPLACE_AREA']:
            final_agg[col]=['min','max']
        elif col in ['CNT_PAYMENT']:
            final_agg[col]=['min','max','mean']
        elif col not in ['SK_ID_CURR']:
            final_agg[col]=['mean','median']
    final_agg_df = joined_prev_app_df.groupby(['SK_ID_CURR']).agg(final_agg)
    final_agg_df.columns = pd.Index([e[0] + "_" + e[1].upper() for e in final_agg_df.columns.tolist()])
    return final_agg_df

def model_creation2(df):
    target_1 = df[df['TARGET'] == 0]
    target_1_sample = target_1.sample(n=int(len(df[df['TARGET']==1])*1.2), random_state=42)
    non_target_1 = df[df['TARGET'] == 1]
    df_filtered = pd.concat([non_target_1, target_1_sample], ignore_index=True)
    df = df_filtered
    df_preprocces_y = df['TARGET'].to_numpy()
    df_preprocces=df.drop(columns=['TARGET'])
    train_x,test_x,train_y,test_y =train_test_split(df_preprocces,df_preprocces_y,test_size=0.3,shuffle=True)
    clf = LGBMClassifier(
            nthread=16,
            n_estimators=10000,
            learning_rate=0.015,
            num_leaves=50,random_state=42)
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (test_x, test_y)], 
            eval_metric= 'auc')
   # clf = MLPClassifier(random_state=42,hidden_layer_sizes=100,activation='relu',solver='sgd',learning_rate='adaptive',shuffle=True,verbose=True,max_iter=10000)
    #clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    y_prob = clf.predict_proba(test_x)[:,1]
    print(classification_report(test_y, y_pred))
    print("ROC AUC:", roc_auc_score(test_y, y_prob))

def main():
    
    main_df = get_applicant_data()
    bureau_df = get_bureau_data()
    prev_app_df = get_prev_application_data()
    
    main_df_joined = main_df.join(bureau_df,on=['SK_ID_CURR'],how='left',rsuffix='_bureau')
    main_df_joined = main_df_joined.join(prev_app_df,on=['SK_ID_CURR'],how='left',rsuffix='_bureau')
    main_df_joined = fillMissing(main_df_joined,True)
    main_df_joined,_ = one_hot_encoder(main_df_joined)
    primary_df =main_df_joined[main_df_joined['TARGET'].notna()]
    eval_df = main_df_joined[main_df_joined['TARGET'].isna()]
    primary_df = primary_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    

    model = model_creation2(primary_df)

    #DEBUG
    # main_df = fillMissing(main_df,True)
    # main_df,_ = one_hot_encoder(main_df)
    # primary_df =main_df[main_df['TARGET'].notna()]
    # primary_df = primary_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    # model = model_creation2(primary_df)



if __name__=="__main__":
    main()