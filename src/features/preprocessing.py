import multiprocessing
import os
import glob
import pandas as pd
from tqdm import tqdm
# import h5py
# import tables
import constants
from datetime import datetime, timedelta
import numpy as np

data = []
main = ""

def genCSVForAnalysis(file):
    # PARSING INDIVIDUAL DATASETS
    if 'chase' in file.lower():
        study = '1'
    elif 'buckingham' in file.lower():
        study = '2'
    elif 'weinstock' in file.lower():
        study = '4'
    elif 'tamborlane' in file.lower():
        study = '3'
    elif 'aleppo' in file.lower():
        study = '5'
    else:
        study = '6'

    df = pd.read_csv(file)
    df.loc[df["gl"] < 20, "gl"] = 0
    df['gl'] = df['gl'].replace({'0':np.nan, 0:np.nan})
    print(f'will drop {df.time.isna().sum()} because there is no timestamp for these values')
    df = df.dropna(subset=['time'])
    print(f'will drop {df.gl.isna().sum()} because there is no glucose level for these values')
    df = df.dropna(subset=['gl'])
    print("read the csv")
    df.time = pd.to_datetime(df.time, errors='coerce')
    # df['t'] = df.time.dt.time
    df.time = (df.time - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    df.sort_values(by=['id', 'time'], ascending=[True, True])
    print("set time as an index value")
    # df['delta'] = df.time.diff().dt.total_seconds().div(60)
    df['delta'] = df.time.diff().div(60)
    df.delta = df.delta.fillna(0)
    # df['normaltime'] = pd.to_datetime(df['time'],unit='s',origin='unix')
    print("created a delta")
    df['studyNum'] = study
    df.id = df.id.astype(str)
    df['id'] = df[['studyNum', 'id']].agg('_'.join, axis=1)
    print("altered the study id")
    print("going to preprocessing")
    return df


def cutFiles(df, dataset_interim_files): # this function is okay
    print("got into cutFiles")
    df.sort_values(by=['id', 'time'], ascending=[True, True])
    for person, person_group_df in tqdm(df.groupby('id'), total=len(df['id'].unique())):
        count = 0
        timeThreshold = 60.0
        for index, row in person_group_df.iterrows():
            if abs(row['delta']) > timeThreshold:
                count += 1
            person_group_df.loc[index, 'dataCount'] = count

        for data_count, count_group_df in person_group_df.groupby('dataCount'):
            # print(person)
            count_group_df.to_csv(f"{dataset_interim_files}{person}_{data_count}.csv")

    print("done cutting the files")


def parse(df, dataset_processed_file): # this function hasn't changed
    for person, grouped_df in df.groupby(['id','dataCount']):
        parseTimeseries(grouped_df, dataset_processed_file)


def parseTimeseries(df, fileout):
    data_event = {}
    prevValue = 0
    prevIndex = 0
    rising = False
    falling = False
    minBGL = 1000
    # window = 2
    countRises = 0    
    risingFlag = False

    df = df.sort_index()
    # if you want to do a simple moving average on the data
    # df.gl = df.gl.rolling(window=window).mean()
    # df = df.tail(df.shape[0] - (window - 1))
    

    for index, row in df.iterrows():
        if index == df.index[-1]:
            ''' can remove all data events that don't hit the end of the file'''
            # if row['gl'] < minBGL:
            #     minBGL = row['gl']
            #     data_event['minBGL'] = minBGL
            #     data_event['global_min_time'] = index

            # if 'id' not in data_event:          #JL writes patient id (patient id + measurement period id) to data_event
            #     data_event['id'] = row['id']    #JL

            # if ('start_time_pre_hypo' in data_event and 'start_time_hypo' not in data_event) or ('start_time_pre_hypo' not in data_event): #JL
            #     # data_event.pop('start_time_pre_hypo', None)
            #     # data_event.pop('start_time_pre_hypo_gl', None)
            #     pass
            # else:
            #     # WE CAN REMOVE THIS. THERE ARE NO MOMENTS WHERE THE SCRIPT GETS TO THE END OF THE FILE AND INCLUDES THE END OF HYPO
            #     data_event['end_of_file_time'] = index
            #     data_event['end_of_file_time_gl'] = row['gl']
            #     data.append(data_event)             #appends current values of 'data_event' into 'data'
            ''' comment up to here for just outputting events that don't hit the end of the file'''
            data_event = {}                     #clears 'data_event' for next iteration
            minBGL = 1000                       #resets 'minBGL' for next iteration
        else:                                   #for all rows except the last
            if prevValue < row['gl']:
                rising = True
                falling = False
            elif prevValue > row['gl']:         #initial iteration is always rising since prevValue initialized to 0
                rising = False
                falling = True
            else:
                rising = False
                falling = False

            thresholdHypo = row['gl'] < 70
            thresholdSevere = row['gl'] <= 54

            if 'id' not in data_event:          #writes patient id (patient id + measurement period id) to data_event
                data_event['id'] = row['id']

            if row['gl'] < minBGL:
                minBGL = row['gl']
                data_event['minBGL'] = minBGL
                data_event['global_min_time'] = index
            else:
                pass

            if not thresholdHypo:
                # start hypo or start severe or (start pre and start hypo)
                if 'start_time_hypo' in data_event or 'start_time_severe' in data_event: # or \
                        # ('start_time_pre_hypo' in data_event and 'start_time_hypo' in data_event):
                    data_event['end_hypo_time'] = index
                    data_event['end_hypo_time_gl'] = row['gl']

                if 'end_hypo_time' in data_event:
                    data_event['minBGL'] = minBGL
                    data.append(data_event)
                    data_event = {}
                    minBGL = 1000

                if not rising and 'start_time_pre_hypo' not in data_event:     #CB
                    data_event['start_time_pre_hypo'] = prevIndex
                    data_event['start_time_pre_hypo_gl'] = prevValue
                    risingFlag = False

                if not rising and 'start_time_pre_hypo' in data_event and risingFlag:
                    data_event['start_time_pre_hypo'] = prevIndex
                    data_event['start_time_pre_hypo_gl'] = prevValue
                    risingFlag = False

                if rising and 'start_time_pre_hypo' in data_event and 'start_time_hypo' not in data_event:
                    risingFlag = True
                else:
                    pass

            # collect hypo data
            if thresholdHypo and not thresholdSevere and 'start_time_severe' not in data_event:
                if 'start_time_hypo' not in data_event:
                    data_event['start_time_hypo'] = index
                    data_event['start_time_hypo_gl'] = row['gl']

            # collect data if goes into severe hypo
            if thresholdHypo and thresholdSevere:
                if 'start_time_severe' not in data_event:
                    data_event['start_time_severe'] = index
                    data_event['start_time_severe_gl'] = row['gl']
                
                if 'start_time_hypo' not in data_event:               #JL
                    data_event['start_time_hypo'] = index             #JL
                    data_event['start_time_hypo_gl'] = row['gl']      #JL

            if thresholdHypo:
                if 'start_time_pre_hypo' not in data_event:
                    data_event['start_time_pre_hypo'] = prevIndex
                    data_event['start_time_pre_hypo_gl'] = prevValue
                if rising and 'last_full_rise' not in data_event:
                    data_event['last_full_rise'] = index
                    data_event['last_full_rise_gl'] = row['gl']
                if rising and f'initial_rise_{countRises}' not in data_event:
                    data_event[f'initial_rise_{countRises}'] = index
                    data_event[f'initial_rise_gl_{countRises}'] = row['gl']
                if f'initial_rise_{countRises}' in data_event and f'treatment_20_{countRises}' not in data_event:
                    duration_between_now_and_initial_rise = index - data_event[f'initial_rise_{countRises}']
                    difference_between_current_gl_and_initial_rise_gl = row['gl'] - data_event[f'initial_rise_gl_{countRises}']
                    # if the duration is less than 45 mins and the gls are over 20 mg/dL, then a treatment event triggers
                    if (duration_between_now_and_initial_rise.total_seconds() <= 45.0*60.0) and (difference_between_current_gl_and_initial_rise_gl > 20):
                        data_event[f'treatment_20_{countRises}'] = index
                        data_event[f'treatment_20_gl_{countRises}'] = row['gl']
                        countRises += 1

            prevValue = row['gl']
            prevIndex = index

    df = pd.DataFrame(data)
    df.to_csv(fileout, sep=',', encoding='utf-8')


def cleanDataset(file, dataset_processed_folder):
    df = pd.read_csv(file, sep=",")
    df = df.drop(columns='Unnamed: 0')

    # capture if the signal gets cut off at the end
    df['start_time_pre_hypo'] = df['start_time_pre_hypo'].replace({'0':np.nan, 0:np.nan})
    data = df.dropna(subset=['start_time_hypo', 'start_time_pre_hypo'])
    data = data.reset_index(drop=True)
    # jl
    no_dropped = df.id.count() - data.id.count()
    print(f"{no_dropped} dropped due to Null values")

    pre_drop = data.id.count()
    data = data.drop_duplicates()
    data = data.reset_index(drop=True)
    # jl
    no_dropped = data.id.count() - pre_drop
    print(f"{no_dropped} dropped due to duplicates")

    # filter out the columns that need to be defragmented, clean them and resave new dataframes
    treatmentCols = data.filter(regex='treatment')
    treatmentCols = treatmentCols.replace('NaN', np.nan)
    treatmentCols = treatmentCols.dropna(how='all')
    idx_treatment = pd.isna(treatmentCols.values).argsort(axis=1)
    treatmentCols = pd.DataFrame(treatmentCols.values[np.arange(treatmentCols.shape[0])[:,None], idx_treatment],
                                 index=treatmentCols.index,
                                 columns=treatmentCols.columns,
                                 )
    treatmentCols = treatmentCols.dropna(how='all',axis='columns')

    initialCols = data.filter(regex='initial_rise')
    initialCols = initialCols.replace('NaN', np.nan)
    initialCols = initialCols.dropna(how='all')
    idx_init = pd.isna(initialCols.values).argsort(axis=1)
    initialCols = pd.DataFrame(initialCols.values[np.arange(initialCols.shape[0])[:,None], idx_init],
                                 index=initialCols.index,
                                 columns=initialCols.columns,
                                 )
    initialCols = initialCols.dropna(how='all', axis='columns')

    df_noTreatment = data[data.columns.drop(list(data.filter(regex=r'(treatment|initial_rise)')))]
    treatedCols = initialCols.join(treatmentCols, how='outer')
    treatment = df_noTreatment.join(treatedCols)

    # writing out if they're multiple events within an hour
    print("gets to subtracting time")
    treatment["start_time_pre_hypo"] = treatment["start_time_pre_hypo"].str.split(".").str[0]
    treatment["end_hypo_time"] = treatment["end_hypo_time"].str.split(".").str[0]
    treatment.start_time_pre_hypo = pd.to_datetime(treatment.start_time_pre_hypo, format="%Y-%m-%d %H:%M:%S.%f", errors='coerce')
    treatment.end_hypo_time = pd.to_datetime(treatment.end_hypo_time, format="%Y-%m-%d %H:%M:%S.%f", errors='coerce')

    # print(treatment)
    treatment = treatment.reset_index(drop=True)
    treatment['new_count'] = 0
    treatment['event_id'] = 0
    treatment = treatment.sort_values(by=['start_time_hypo', 'id'])
    event_id = 0
    

    for event, grouped_event in treatment.groupby(['id']):
        # print("new group")
        grouped_event = grouped_event.sort_values(by=['start_time_hypo'])
        prev_end_time = "1870-01-01 00:00:01"
        new_group = 0
        new_count = 0
        old_id = '0'

        for idx, row in grouped_event.iterrows():
            starttime = datetime.strptime(str(row.start_time_pre_hypo), "%Y-%m-%d %H:%M:%S")
            endtime = datetime.strptime(str(prev_end_time), "%Y-%m-%d %H:%M:%S")
            if new_group == 0:
                if (old_id == row.id):
                    treatment.loc[idx, 'event_id'] = event_id
                    treatment.loc[idx, 'new_count'] = new_count
                    event_id = event_id + 1
                    prev_end_time = row.end_hypo_time
                    new_group = 1
                    old_id = row.id
                else:
                    event_id = event_id + 1
                    treatment.loc[idx, 'event_id'] = event_id
                    treatment.loc[idx, 'new_count'] = new_count
                    prev_end_time = row.end_hypo_time
                    new_group = 1
                    old_id = row.id
            else:
                if (old_id == row.id) and (starttime - endtime) <= timedelta(hours=1):
                    new_count = new_count + 1
                    treatment.loc[idx, 'event_id'] = event_id
                    treatment.loc[idx, 'new_count'] = new_count
                    prev_end_time = row.end_hypo_time
                    old_id = row.id
                else: 
                    event_id = event_id + 1
                    new_count = 0
                    treatment.loc[idx, 'event_id'] = event_id
                    treatment.loc[idx, 'new_count'] = new_count
                    prev_end_time = row.end_hypo_time
                    old_id = row.id

    # treatment = treatment.sort_values(by=['event_id', 'start_time_hypo'])
    # treatment = treatment.drop(columns=['last_full_rise', 'last_full_rise_gl', 'global_min_time', 'minBGL', 'initial_rise_0', 'initial_rise_gl_0', 'initial_rise_1', 'initial_rise_gl_1', \
    #     'treatment_20_0', 'treatment_20_gl_0', 'treatment_20_1', 'treatment_20_gl_1', 'start_time_hypo_gl', 'start_time_severe_gl', 'end_hypo_time_gl', 'start_time_pre_hypo_gl'])

    print('== parsed data ==')
    print('all events in data:', df['id'].count())
    print('number of omitted events that start during hypo:', len(df[df.start_time_hypo.notna() & df.start_time_pre_hypo.isna()]))
    print('number of omitted events that end during hypo:', len(df[df.start_time_hypo.notna() & df.end_hypo_time.isna()]))
    # print('number of omitted events that start and end during hypo:',
        #   len(df[df.start_time_hypo.notna() & df.start_time_pre_hypo.isna() & df.end_hypo_time.isna()]))
    # print('number of events that end during hypo (based on gl):', len(df[df.end_of_file_time_gl < 70]))
    # print('number of end-of-file events:', len(df[df.start_time_hypo.isna() & df.end_of_file_time.notna()]))
    # print('number of end-of-file events (based on gl):', len(df[df.end_of_file_time_gl > 70]))
    print('number of complete events:', len(treatment))
    print('== end ==')

    # treatment = treatment.drop(columns=['dur_between_start(curr)_end(prev)'])
    # print(f'{dataset_processed_folder}{file.split("/")[-1].split(".")[0]}_cleaned.csv')
    treatment.to_csv(f'{dataset_processed_folder}{file.split("/")[-1].split(".")[0]}_cleaned.csv')
    print('finished cleaning the data')

# all_csv_files = [file
#                      for path, subdir, files in os.walk(dataset_interim_combined_dir)
#                      for file in glob(os.path.join(dataset_interim_combined_dir, extension))]

def calcDurations(file): # this function has changed
    filename = file
    print(filename)
    file = pd.read_csv(file)
    file.start_time = pd.to_datetime(file.start_time, errors='coerce')
    file.end_time = pd.to_datetime(file.end_time, errors='coerce')
    file['durations'] = file.end_time - file.start_time
    file.to_csv(filename)

    # JL's code for returning the duration of each file
    # total_duration = file['durations'].sum()
    # print(total_duration)

    # return(file, total_duration)


def processForVis(file, dataset_interim_individual_folder, dataset_acceptable_files): # this function hasn't changed

    dateset_name = file.split('/')[-1].split('_')[-2].split('.')[0]
    print(dateset_name)
    df_timestampsFile = pd.read_csv(file, sep=",")
    # df_timestampsFile = df_timestampsFile.sort_values(by=['id', 'start_time_pre_hypo'], ascending=[True, True])
    df_timestampsFile = df_timestampsFile.drop(columns='Unnamed: 0')
    df_timestampsFile = df_timestampsFile.drop_duplicates()
    df_timestampsFile = df_timestampsFile.reset_index(drop=True)
    df_timestampsFile.start_time_pre_hypo = pd.to_datetime(df_timestampsFile.start_time_pre_hypo)
    df_timestampsFile.end_hypo_time = pd.to_datetime(df_timestampsFile.end_hypo_time)

    filepath = f'../KineticsOfHypoglycemiaAnalysis/data/interim/individualData/{dateset_name}'

    open(f'../KineticsOfHypoglycemiaAnalysis/data/processed/visData/visualizationData_{dateset_name}.csv', mode='w').close()

    flag_more_than_one = False
    previous_masked_data = pd.DataFrame()
    previous_file = ''
    newWrite = True
    # print(all_csv_files)

    # STEP 2B: initial preprocessing
    with open(dataset_acceptable_files,'r') as fd:
        all_csv_files = [filename.rstrip() for filename in fd.readlines()]
    eventId = 0
    new_count = 0
    for index, row in tqdm(df_timestampsFile.iterrows(), total=df_timestampsFile.shape[0]):
        eventId = row.event_id
        new_count = row.new_count
        # print(row.start_time_pre_hypo)
        for files in all_csv_files:
            if '_'.join(files.split('/')[-1].split('_',2)[:2]) == str(row['id']):
            # if '_'.join(files.split('_',2)[:2]) == str(row['id']):
                # df_rawFiles = pd.read_csv(f'{filepath}/{files}')
                df_rawFiles = pd.read_csv(files)
                df_rawFiles.time = df_rawFiles.time.astype(str)
                df_rawFiles['event_id'] = eventId
                df_rawFiles['new_count'] = new_count
                # print(df_rawFiles.time)
                # print(row.start_time_pre_hypo)

                if df_rawFiles['time'].str.contains(str(row.start_time_pre_hypo)).any() and newWrite:
                    # print("new file")
                    # print(f'new file: {files}')
                    mask = (df_rawFiles.time >= str(row.start_time_pre_hypo)) & (df_rawFiles.time <= str(row.end_hypo_time))
                    previous_masked_data = df_rawFiles.loc[mask]
                    newWrite = False
                    flag_more_than_one = True
                    # print(row.start_time_pre_hypo)
                    # print(df_rawFiles)

                elif df_rawFiles.time.str.contains(str(row.start_time_pre_hypo)).any() and not newWrite:
                    if flag_more_than_one:
                        mask = (df_rawFiles.time >= str(row.start_time_pre_hypo)) & (df_rawFiles.time <= str(row.end_hypo_time))
                        current_masked_data = df_rawFiles.loc[mask]
                        if (previous_masked_data.iloc[0,2] == current_masked_data.iloc[0,2]) and (previous_masked_data.iloc[0,3] == current_masked_data.iloc[0,3]):
                        # check if the first two rows are similar. if they are then
                            if len(current_masked_data) > len(previous_masked_data):
                                # print("duplicate where current is greater than previous")
                                # print(f'file: {files}')
                                current_masked_data.to_csv(f'../KineticsOfHypoglycemiaAnalysis/data/processed/visData/visualizationData_{dateset_name}.csv', mode='a')
                                flag_more_than_one = False
                                previous_masked_data = current_masked_data
                            elif len(current_masked_data) < len(previous_masked_data):
                                # print("duplicate where previous is greater than current")
                                # print(f'file: {files}')

                                previous_masked_data.to_csv(f'../KineticsOfHypoglycemiaAnalysis/data/processed/visData/visualizationData_{dateset_name}.csv', mode='a')
                                flag_more_than_one = False
                                previous_masked_data = current_masked_data
                            else:
                                # print("duplicate where they are equal")
                                # print(f'file: {files}')
                                current_masked_data.to_csv(f'../KineticsOfHypoglycemiaAnalysis/data/processed/visData/visualizationData_{dateset_name}.csv', mode='a')
                                flag_more_than_one = False
                                previous_masked_data = current_masked_data
                        # then write out the previous one and set the current one to previous
                        else:
                            # print("there's more than one, but it doesn't equal the first")
                            # print(f'file: {files}')
                            previous_masked_data.to_csv(f'../KineticsOfHypoglycemiaAnalysis/data/processed/visData/visualizationData_{dateset_name}.csv', mode='a')
                            flag_more_than_one = True
                            previous_masked_data = current_masked_data
                    else:
                        mask = (df_rawFiles.time >= str(row.start_time_pre_hypo)) & (df_rawFiles.time <= str(row.end_hypo_time))
                        previous_masked_data = df_rawFiles.loc[mask]
                        newWrite = False
                        flag_more_than_one = True
                else:
                    # print("not a useful file")
                    pass



