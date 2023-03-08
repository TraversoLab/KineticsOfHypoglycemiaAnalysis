from lib2to3.pgen2.pgen import generate_grammar
from tokenize import group
import numpy as np
import pandas as pd
from scipy.stats import iqr

from visualization import plottingData as vs
import datetime
import os
from glob import glob

from datetime import datetime
from datetime import timedelta
import warnings

longTimeInHypo = []


enrollmentDuration = 182
enrollmentDuration_weinstock = 14
enrollmentDuration_buckingham = 91

def parseWholeEvents(whole_hypo_events):

    # get data from both day and night
    whole_hypo_events.start_time_hypo = pd.to_datetime(whole_hypo_events.start_time_hypo, errors='coerce')
    whole_hypo_events = whole_hypo_events[whole_hypo_events['end_hypo_time'].notna()]
    whole_hypo_events = whole_hypo_events.set_index('start_time_hypo', drop=False)
    whole_hypo_events = whole_hypo_events.drop(columns=['Unnamed: 0'])
    whole_hypo_events['tod'] = whole_hypo_events.index.isin(whole_hypo_events.between_time('23:00:00', '06:00:00', include_start=True, include_end=False).index)
    whole_hypo_events['tod'] = whole_hypo_events['tod'].astype(int) # false = 0 (day); true = 1 (night)

    ''' to use if you want to just separate all individual events by time'''
    # df_night = whole_hypo_events.between_time('23:00:00', '05:59:59')
    # df_day = whole_hypo_events.between_time('06:00:00', '22:59:59')
    ''' ~~~~~ '''

    ''' to use if you're calculating events based on consecutive sub-events '''
    col_names = ['id', 'minBGL', 'global_min_time', 'start_time_hypo', 'start_time_hypo_gl', 'start_time_pre_hypo', 'start_time_pre_hypo_gl',
        'last_full_rise', 'last_full_rise_gl', 'end_hypo_time', 'end_hypo_time_gl', 'start_time_severe', 'start_time_severe_gl',
        'initial_rise_0', 'initial_rise_gl_0', 'initial_rise_1', 'initial_rise_gl_1', 'treatment_20_0', 'treatment_20_gl_0', 'treatment_20_1', 'treatment_20_gl_1', 'new_count', 'event_id', 'tod']
    df_day = pd.DataFrame(columns=col_names)
    df_night = pd.DataFrame(columns=col_names)

    '''
    all events = all events including those are beign kicked out //
    complete events = all events not including being kicked out //
    initial event = all "new_count" events = 0 // 
    followup events = everything greater than 0 //
    number of child events = "new_count" = number // 
    '''

    whole_hypo_events['numSubeventsInParent'] = whole_hypo_events.groupby('event_id')['new_count'].transform('size')

    whole_hypo_events = whole_hypo_events.reset_index(drop=True)
    print(f'the length of all hypo events are outside the if statement are : {len(whole_hypo_events)}')
    for event, grouped_event in whole_hypo_events.groupby(['event_id']):
        # with the grouped_events, we should get the start time of the first event
        for index, row in grouped_event.iterrows():
            if row.new_count == 0 and row.tod == 0:
                df_day = df_day.append(grouped_event).copy()
                break
            elif row.new_count != 0:
                break
            else:
                df_night = df_night.append(grouped_event).copy()
                break
    ''' ~~~~~ '''
    # print(df_day)
    # df_day.to_csv(f'../KineticsOfHypoglycemiaAnalysis/data/processed/df_day_all.csv')
    # df_night.to_csv(f'../KineticsOfHypoglycemiaAnalysis/data/processed/df_night_all.csv')

    return df_day, df_night


def prepFilesForStats(a):
    # whole_hypo_events = pd.DataFrame()
    ''' THIS BASICALLY CONVERTS AN ANALYZED DATASET TO DAYTIME AND NIGHTTIME DATASETS'''
    warnings.filterwarnings("ignore")
    if isinstance(a, list):
        print(a)
        all_csv_files = []
        for t in a:
            filename = f'../KineticsOfHypoglycemiaAnalysis/data/processed/cleanedData_noAleppo/data_{t}_cleaned.csv'
            all_csv_files.append(filename)
        
        data_day = []
        data_night = []
        inc = 100000
        for file in all_csv_files:
            df = pd.read_csv(file, sep=",")
            df['event_id'] = df['event_id'] + inc
            day, night = parseWholeEvents(df)
            data_day.append(day)
            data_night.append(night)
            inc = inc + 100000

        df_day = pd.concat(data_day, ignore_index=True)
        df_night = pd.concat(data_night, ignore_index=True)

        print(f'Number of daytime events: {len(df_day)}')
        print(f'Number of nighttime events: {len(df_night)}' + '\n')
        # this should be anything = 0 and then anything greater than 0 and make all greater than equal to
        print(f'Number of complete events: {(df_day["new_count"] == 0).sum() + (df_night["new_count"] == 0).sum()}')
        print(f'Number of initial events = 0: {((df_day["new_count"] == 0) & (df_day["numSubeventsInParent"] >= 1)).sum() + ((df_night["new_count"] == 0) & (df_night["numSubeventsInParent"] >= 1)).sum()}')
        print(f'Number of follow-up events > 1: {((df_day["new_count"] >= 1) & (df_day["numSubeventsInParent"] >= 2)).sum() + ((df_night["new_count"] >= 1) & (df_night["numSubeventsInParent"] >= 2)).sum()}')
        print(f'Number of complete parent events with 1: {((df_day["new_count"] == 1) & (df_day["numSubeventsInParent"] == 2)).sum() + ((df_night["new_count"] == 1) & (df_night["numSubeventsInParent"] == 2)).sum()}')
        print(f'Number of complete parent events with 2: {((df_day["new_count"] == 2) & (df_day["numSubeventsInParent"] == 3)).sum() + ((df_night["new_count"] == 2) & (df_night["numSubeventsInParent"] == 3)).sum()}')
        print(f'Number of complete parent events with 3: {((df_day["new_count"] == 3) & (df_day["numSubeventsInParent"] == 4)).sum() + ((df_night["new_count"] == 3) & (df_night["numSubeventsInParent"] == 4)).sum()}')
        print(f'Number of complete parent events with 4: {((df_day["new_count"] == 4) & (df_day["numSubeventsInParent"] == 5)).sum() + ((df_night["new_count"] == 4) & (df_night["numSubeventsInParent"] == 5)).sum()}')
        print(f'Number of complete parent events with 5: {((df_day["new_count"] == 5) & (df_day["numSubeventsInParent"] == 6)).sum() + ((df_night["new_count"] == 5) & (df_night["numSubeventsInParent"] == 6)).sum()}')
        print(f'Number of complete parent events with 5+: {((df_day["new_count"] > 5) & (df_day["numSubeventsInParent"] > 6)).sum() + ((df_night["new_count"] > 5) & (df_night["numSubeventsInParent"] > 6)).sum()}')

        alldata_durations = timedelta(hours=1)
        alldata_durations_day = timedelta(hours=1)
        alldata_durations_night = timedelta(hours=1)
        alldata_num_participants = 0
        all_duration_csv_files = []
        for t in a:
            filename = f'../KineticsOfHypoglycemiaAnalysis/data/processed/cleanedData_noAleppo/data_{t}_cleaned.csv'
            # filename = f'../KineticsOfHypoglycemiaAnalysis/data/processed/cleanedData/data_{t}_cleaned.csv'
            all_duration_csv_files.append(filename)

        duration_dir = "../KineticsOfHypoglycemiaAnalysis/data/interim/individualData/durations_noaleppo/"
        # duration_dir = "../KineticsOfHypoglycemiaAnalysis/data/interim/individualData/durations/"
        extention = "*.csv"
        all_csv_files_dur = [file
                        for path, subdir, files in os.walk(duration_dir)
                        for file in glob(os.path.join(duration_dir, extention))]

        for file in all_csv_files_dur:
            df = pd.read_csv(file, sep=",")
            alldata_durations_df = pd.read_csv(file)
            alldata_durations_df.start_time = pd.to_datetime(alldata_durations_df.start_time, errors='coerce')
            alldata_durations_df.end_time = pd.to_datetime(alldata_durations_df.end_time, errors='coerce')
            alldata_durations_df['durations'] = alldata_durations_df.end_time - alldata_durations_df.start_time
            alldata_durations_ind = abs(alldata_durations_df.durations.sum())
            alldata_durations += alldata_durations_ind
            alldata_durations_df = alldata_durations_df.set_index('start_time', drop=False)
            alldata_num_participants += len(pd.unique(alldata_durations_df['studyid']))
            alldata_day = alldata_durations_df.between_time('06:00:00', '22:59:59', inclusive="right")
            alldata_durations_day_ind = abs(alldata_day.durations.sum())
            alldata_durations_day += alldata_durations_day_ind
            alldata_night = alldata_durations_df.between_time('23:00:00', '05:59:59', inclusive="right")
            alldata_durations_night_ind = abs(alldata_night.durations.sum())
            alldata_durations_night += alldata_durations_night_ind
            # alldata_patientyear = alldata_durations_df.groupby('studyid')['durations'].sum()
        
        alldata_patientyear = alldata_durations / alldata_num_participants


        # print(alldata_durations)


        print(f'Durations : {alldata_durations}')
        print(f'Number of participants : {alldata_num_participants}')
        print(f'Number of days of data per participant : {(alldata_durations)/alldata_num_participants}')
        print(f'Durations Day : {alldata_durations_day}')
        print(f'Durations Night : {alldata_durations_night}')

    elif isinstance(a, str):
        print(a)
        file = f'../KineticsOfHypoglycemiaAnalysis/data/processed/cleanedData/data_{a}_cleaned.csv'
        print(file)
        whole_hypo_events = pd.read_csv(file)
        # print(f'Number of initial events = 0: {(whole_hypo_events["new_count"] == 0).sum()}')
        # print(f'Number of follow-up events < 1: {(whole_hypo_events["new_count"] >= 1).sum()}')
        # print(f'Number of complete parent events with 1: {(whole_hypo_events["new_count"] == 1).sum()}')
        # print(f'Number of complete parent events with 2: {(whole_hypo_events["new_count"] == 2).sum()}')
        # print(f'Number of complete parent events with 3: {(whole_hypo_events["new_count"] == 3).sum()}')
        # print(f'Number of complete parent events with 4: {(whole_hypo_events["new_count"] == 4).sum()}')
        # print(f'Number of complete parent events with 5: {(whole_hypo_events["new_count"] == 5).sum()}')
        # print(f'Number of complete parent events with 5+: {(whole_hypo_events["new_count"] > 5).sum()}')
        df_day, df_night = parseWholeEvents(whole_hypo_events)

        duration_file = f'../KineticsOfHypoglycemiaAnalysis/data/interim/individualData/durations/{a}_all_files_durations.csv'
        alldata_durations_df = pd.read_csv(duration_file)
        alldata_durations_df.start_time = pd.to_datetime(alldata_durations_df.start_time, errors='coerce')
        alldata_durations_df.end_time = pd.to_datetime(alldata_durations_df.end_time, errors='coerce')
        alldata_durations_df['durations'] = alldata_durations_df.end_time - alldata_durations_df.start_time
        alldata_durations = abs(alldata_durations_df.durations.sum())
        alldata_num_participants = len(pd.unique(alldata_durations_df['studyid']))
        # alldata_patientyear = alldata_durations_df.groupby('studyid')['durations'].sum()
        alldata_patientyear = alldata_durations / alldata_num_participants

        # alldata_durations_df = alldata_durations_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
        alldata_durations_df = alldata_durations_df.drop(columns=['Unnamed: 0'])
        alldata_durations_df = alldata_durations_df.set_index('start_time', drop=False)
        alldata_day = alldata_durations_df.between_time('06:00:00', '22:59:59', inclusive="right")
        alldata_durations_day = abs(alldata_day.durations.sum())
        alldata_night = alldata_durations_df.between_time('23:00:00', '05:59:59', inclusive="right")
        alldata_durations_night = abs(alldata_night.durations.sum())

        # this should be anything = 0 and then anything greater than 0 and make all greater than equal to
        print(f'Number of complete events: {len(df_day) + len(df_night)}')
        print(f'Number of parent events: {(df_day["new_count"] == 0).sum() + (df_night["new_count"] == 0).sum()}')
        print(f'Number of initial events = 0: {((df_day["new_count"] == 0) & (df_day["numSubeventsInParent"] >= 1)).sum() + ((df_night["new_count"] == 0) & (df_night["numSubeventsInParent"] >= 1)).sum()}')
        print(f'Number of follow-up events > 1: {((df_day["new_count"] >= 1) & (df_day["numSubeventsInParent"] >= 2)).sum() + ((df_night["new_count"] >= 1) & (df_night["numSubeventsInParent"] >= 2)).sum()}')
        print(f'Number of parent events = 0 and no follow-on events: {((df_day["new_count"] == 0) & (df_day["numSubeventsInParent"] == 1)).sum() + ((df_night["new_count"] == 0) & (df_night["numSubeventsInParent"] == 1)).sum()}')
        print(f'Number of complete parent events with 1: {((df_day["new_count"] == 1) & (df_day["numSubeventsInParent"] == 2)).sum() + ((df_night["new_count"] == 1) & (df_night["numSubeventsInParent"] == 2)).sum()}')
        print(f'Number of complete parent events with 2: {((df_day["new_count"] == 2) & (df_day["numSubeventsInParent"] == 3)).sum() + ((df_night["new_count"] == 2) & (df_night["numSubeventsInParent"] == 3)).sum()}')
        print(f'Number of complete parent events with 3: {((df_day["new_count"] == 3) & (df_day["numSubeventsInParent"] == 4)).sum() + ((df_night["new_count"] == 3) & (df_night["numSubeventsInParent"] == 4)).sum()}')
        print(f'Number of complete parent events with 4: {((df_day["new_count"] == 4) & (df_day["numSubeventsInParent"] == 5)).sum() + ((df_night["new_count"] == 4) & (df_night["numSubeventsInParent"] == 5)).sum()}')
        print(f'Number of complete parent events with 5: {((df_day["new_count"] == 5) & (df_day["numSubeventsInParent"] == 6)).sum() + ((df_night["new_count"] == 5) & (df_night["numSubeventsInParent"] == 6)).sum()}')
        print(f'Number of complete parent events with 5+: {((df_day["new_count"] > 6) & (df_day["numSubeventsInParent"] > 6)).sum() + ((df_night["new_count"] > 5) & (df_night["numSubeventsInParent"] > 6)).sum()}')

        print(f'Number of daytime events: {len(df_day)}')
        print(f'Number of nighttime events: {len(df_night)}' + '\n')

        print(f'Durations : {alldata_durations}')
        print(f'Number of participants : {alldata_num_participants}')
        print(f'Number of days of data per participant : {(alldata_durations)/alldata_num_participants}')
        print(f'Durations Day : {alldata_durations_day}')
        print(f'Durations Night : {alldata_durations_night}')

    else:
        raise Exception("Don't know what to do here.")
    
    # df_day.to_csv(f'../KineticsOfHypoglycemiaAnalysis/data/processed/df_day_{a}.csv')
    # df_night.to_csv(f'../KineticsOfHypoglycemiaAnalysis/data/processed/df_night_{a}.csv')
    return df_day, df_night, abs(alldata_durations), alldata_num_participants, alldata_patientyear


def getStats(df_day, df_night, alldata_durations, alldata_num_participants, alldata_patientyear, author):

    # print(df_day)
    # separate into daytime mild, nighttime mild, daytime severe, nighttime
    day_mild, day_severe = splitGroupedEventsIntoMildAndSevereData(df_day)
    night_mild, night_severe = splitGroupedEventsIntoMildAndSevereData(df_night)
    day_mild, day_severe = statPrep(day_mild, day_severe)
    night_mild, night_severe = statPrep(night_mild, night_severe)
    print(f'Duration of day mild data: {str(timedelta(seconds = day_mild.duration_hypo.sum()))}')
    print(f'Duration of day severe data: {str(timedelta(seconds = day_severe.duration_hypo.sum()))}')
    print(f'Duration of night mild data: {str(timedelta(seconds = night_mild.duration_hypo.sum()))}')
    print(f'Duration of night severe data: {str(timedelta(seconds = night_severe.duration_hypo.sum()))}')
    print(f'Duration of mild data: {str(timedelta(seconds = (day_mild.duration_prehypo_to_end.sum() + night_mild.duration_prehypo_to_end.sum())))}')
    print(f'Duration of severe data: {str(timedelta(seconds = (day_severe.duration_prehypo_to_end.sum() + night_severe.duration_prehypo_to_end.sum())))}')
    print(f'Duration of day data: {str(timedelta(seconds = (day_mild.duration_prehypo_to_end.sum() + day_severe.duration_prehypo_to_end.sum())))}')
    print(f'Duration of night data: {str(timedelta(seconds = (night_mild.duration_prehypo_to_end.sum() + night_severe.duration_prehypo_to_end.sum())))}' + '\n')
    print(f'Duration of day mild data ppy: {day_mild.duration_hypo.sum() / 3600 / alldata_num_participants}')
    print(f'Duration of day severe data ppy: {day_severe.duration_hypo.sum() / 3600 / alldata_num_participants}')
    print(f'Duration of night mild data ppy: {night_mild.duration_hypo.sum() / 3600 / alldata_num_participants}')
    print(f'Duration of night severe data ppy: {night_severe.duration_hypo.sum() / 3600 / alldata_num_participants}')
    print(f'Duration of mild data ppy: {(day_mild.duration_prehypo_to_end.sum() + night_mild.duration_prehypo_to_end.sum()) / 3600 / alldata_num_participants}')
    print(f'Duration of severe data ppy: {(day_severe.duration_prehypo_to_end.sum() + night_severe.duration_prehypo_to_end.sum()) / 3600 / alldata_num_participants}')
    print(f'Duration of day data ppy: {(day_mild.duration_prehypo_to_end.sum() + day_severe.duration_prehypo_to_end.sum()) / 3600 / alldata_num_participants}')
    print(f'Duration of night data ppy: {(night_mild.duration_prehypo_to_end.sum() + night_severe.duration_prehypo_to_end.sum()) / 3600 / alldata_num_participants}')

    # # start analyzing
    print("daytime")
    averages_day = calcStats(day_mild, day_severe, alldata_durations, alldata_num_participants, alldata_patientyear, flag=0)
    
    print("nighttime")
    averages_night = calcStats(night_mild, night_severe, alldata_durations, alldata_num_participants, alldata_patientyear, flag=1)

    data_day = pd.DataFrame.from_dict(averages_day, orient='index').transpose()
    data_night = pd.DataFrame.from_dict(averages_night, orient='index').transpose()

    # print hypo durations
    print(f'average duration of daytime hypo (mild only): {str(data_day.iloc[0, data_day.columns.get_loc("duration_hypo_mild_only")])} +/- {str(data_day.iloc[0, data_day.columns.get_loc("std_hypo_mild_only")])}')
    print(f'median of daytime hypo (mild only): {str(data_day.iloc[0, data_day.columns.get_loc("med_hypo_mild_only")])}')
    print(f'iqr of daytime hypo (mild only): {str(data_day.iloc[0, data_day.columns.get_loc("iqr_hypo_mild_only")])}')
    print(f'average duration of daytime hypo (severe only): {str(data_day.iloc[0, data_day.columns.get_loc("duration_severe")])} +/- {str(data_day.iloc[0, data_day.columns.get_loc("std_severe")])}')
    print(f'median of daytime hypo (severe only): {str(data_day.iloc[0, data_day.columns.get_loc("med_severe")])}')
    print(f'iqr of daytime hypo (severe only): {str(data_day.iloc[0, data_day.columns.get_loc("iqr_severe")])}')
    print(f'average duration of nighttime hypo (mild only): {str(data_night.iloc[0, data_night.columns.get_loc("duration_hypo_mild_only")])} +/- {str(data_night.iloc[0, data_night.columns.get_loc("std_hypo_mild_only")])}')
    print(f'median of nighttime hypo (mild only): {str(data_night.iloc[0, data_night.columns.get_loc("med_hypo_mild_only")])}')
    print(f'iqr of nighttime hypo (mild only): {str(data_night.iloc[0, data_night.columns.get_loc("iqr_hypo_mild_only")])}')
    print(f'average duration of nighttime hypo (severe only): {str(data_night.iloc[0, data_night.columns.get_loc("duration_severe")])} +/- {str(data_night.iloc[0, data_night.columns.get_loc("std_severe")])}')
    print(f'median of nighttime hypo (severe only): {str(data_night.iloc[0, data_night.columns.get_loc("med_severe")])}')
    print(f'iqr of nighttime hypo (severe only): {str(data_night.iloc[0, data_night.columns.get_loc("iqr_severe")])}' + '\n')

    # print pre-hypo duration
    print(f'average duration of daytime pre-hypo to hypo (mild only): {str(data_day.iloc[0, data_day.columns.get_loc("duration_prehypo_to_hypo_mild_only")])} +/- {str(data_day.iloc[0, data_day.columns.get_loc("std_prehypo_to_hypo_mild_only")])}')
    print(f'median of daytime pre-hypo to hypo (mild only): {str(data_day.iloc[0, data_day.columns.get_loc("med_prehypo_to_hypo_mild_only")])}')
    print(f'iqr of daytime pre-hypo to hypo (mild only): {str(data_day.iloc[0, data_day.columns.get_loc("iqr_prehypo_to_hypo_mild_only")])}')
    print(f'average duration of daytime pre-hypo to hypo (severe only): {str(data_day.iloc[0, data_day.columns.get_loc("duration_prehypo_to_hypo_mean_severe")])} +/- {str(data_day.iloc[0, data_day.columns.get_loc("std_prehypo_to_hypo_mean_severe")])}')
    print(f'median of daytime pre-hypo to hypo (severe only): {str(data_day.iloc[0, data_day.columns.get_loc("med_prehypo_to_hypo_mean_severe")])}')
    print(f'iqr of daytime pre-hypo to hypo (severe only): {str(data_day.iloc[0, data_day.columns.get_loc("iqr_prehypo_to_hypo_mean_severe")])}')    
    print(f'average duration of nighttime pre-hypo to hypo (mild only): {str(data_night.iloc[0, data_night.columns.get_loc("duration_prehypo_to_hypo_mild_only")])} +/- {str(data_night.iloc[0, data_night.columns.get_loc("std_prehypo_to_hypo_mild_only")])}')
    print(f'median of nighttime pre-hypo to hypo (mild only): {str(data_night.iloc[0, data_night.columns.get_loc("med_prehypo_to_hypo_mild_only")])}')
    print(f'iqr of nighttime pre-hypo to hypo (mild only): {str(data_night.iloc[0, data_night.columns.get_loc("iqr_prehypo_to_hypo_mild_only")])}')    
    print(f'average duration of nighttime pre-hypo to hypo (severe only): {str(data_night.iloc[0, data_night.columns.get_loc("duration_prehypo_to_hypo_mean_severe")])} +/- {str(data_night.iloc[0, data_night.columns.get_loc("std_prehypo_to_hypo_mean_severe")])}')
    print(f'median of nighttime pre-hypo to hypo (severe only): {str(data_night.iloc[0, data_night.columns.get_loc("med_prehypo_to_hypo_mean_severe")])}')
    print(f'iqr of nighttime pre-hypo to hypo (severe only): {str(data_night.iloc[0, data_night.columns.get_loc("iqr_prehypo_to_hypo_mean_severe")])}' + '\n')

    # print duration to glucagon treatment
    print(f'average duration to first glucagon treatment of daytime (20 mg / dL in 45 mins treatment recovery)  (severe only): {str(data_day.iloc[0, data_day.columns.get_loc("duration_treatment_short_first_severe")])} +/- {str(data_day.iloc[0, data_day.columns.get_loc("std_treatment_short_first_severe")])}')
    print(f'median of first glucagon treatment of daytime (20 mg / dL in 45 mins treatment recovery)  (severe only): {str(data_day.iloc[0, data_day.columns.get_loc("med_treatment_short_first_severe")])}')
    print(f'iqr of first glucagon treatment of daytime (20 mg / dL in 45 mins treatment recovery)  (severe only): {str(data_day.iloc[0, data_day.columns.get_loc("iqr_treatment_short_first_severe")])}')
    print(f'average duration to last glucagon treatment of daytime (20 mg / dL in 45 mins treatment recovery)  (severe only): {str(data_day.iloc[0, data_day.columns.get_loc("duration_treatment_short_severe")])} +/- {str(data_day.iloc[0, data_day.columns.get_loc("std_treatment_short_severe")])}')
    print(f'median of last glucagon treatment of daytime (20 mg / dL in 45 mins treatment recovery)  (severe only): {str(data_day.iloc[0, data_day.columns.get_loc("med_treatment_short_severe")])}')
    print(f'iqr of last glucagon treatment of daytime (20 mg / dL in 45 mins treatment recovery)  (severe only): {str(data_day.iloc[0, data_day.columns.get_loc("iqr_treatment_short_severe")])}')    
    print(f'average duration to first glucagon treatment of nighttime 20 mg / dL in 45 mins treatment recovery (severe only): {str(data_night.iloc[0, data_night.columns.get_loc("duration_treatment_short_first_severe")])} +/- {str(data_night.iloc[0, data_night.columns.get_loc("std_treatment_short_first_severe")])}')
    print(f'median of first glucagon treatment of nighttime 20 mg / dL in 45 mins treatment recovery (severe only): {str(data_night.iloc[0, data_night.columns.get_loc("med_treatment_short_first_severe")])}')
    print(f'iqr of first glucagon treatment of nighttime 20 mg / dL in 45 mins treatment recovery (severe only): {str(data_night.iloc[0, data_night.columns.get_loc("iqr_treatment_short_first_severe")])}')
    print(f'average duration to last glucagon treatment of nighttime 20 mg / dL in 45 mins treatment recovery (severe only): {str(data_night.iloc[0, data_night.columns.get_loc("duration_treatment_short_severe")])} +/- {str(data_night.iloc[0, data_night.columns.get_loc("std_treatment_short_severe")])}')
    print(f'median of last glucagon treatment of nighttime 20 mg / dL in 45 mins treatment recovery (severe only): {str(data_night.iloc[0, data_night.columns.get_loc("med_treatment_short_severe")])}')
    print(f'iqr of last glucagon treatment of nighttime 20 mg / dL in 45 mins treatment recovery (severe only): {str(data_night.iloc[0, data_night.columns.get_loc("iqr_treatment_short_severe")])}' + '\n')

    # print mild-to-severe hypo duration
    print(f'average duration of daytime mild to severe (severe only): {str(data_day.iloc[0, data_day.columns.get_loc("duration_mild_to_severe_first")])} +/- {str(data_day.iloc[0, data_day.columns.get_loc("std_mild_to_severe_first")])}')
    print(f'median of daytime mild to severe (severe only): {str(data_day.iloc[0, data_day.columns.get_loc("med_mild_to_severe_first")])}')
    print(f'iqr of daytime mild to severe (severe only): {str(data_day.iloc[0, data_day.columns.get_loc("iqr_mild_to_severe_first")])}')
    print(f'average duration of nighttime mild to severe (severe only): {str(data_night.iloc[0, data_night.columns.get_loc("duration_mild_to_severe_first")])} +/- {str(data_night.iloc[0, data_night.columns.get_loc("std_mild_to_severe_first")])}')
    print(f'median of daytime mild to severe (severe only): {str(data_night.iloc[0, data_night.columns.get_loc("med_mild_to_severe_first")])}')
    print(f'iqr of daytime mild to severe (severe only): {str(data_night.iloc[0, data_night.columns.get_loc("iqr_mild_to_severe_first")])}' + '\n')

    # print pre-hypo to severe hypo duration
    print(f'average duration of daytime pre-hypo to severe (severe only): {str(data_day.iloc[0, data_day.columns.get_loc("duration_pre_hypo_to_severe")])} +/- {str(data_day.iloc[0, data_day.columns.get_loc("std_pre_hypo_to_severe")])}')
    print(f'median duration of daytime pre-hypo to severe (severe only): {str(data_day.iloc[0, data_day.columns.get_loc("med_pre_hypo_to_severe")])}')
    print(f'iqr duration of daytime pre-hypo to severe (severe only): {str(data_day.iloc[0, data_day.columns.get_loc("iqr_pre_hypo_to_severe")])}')
    print(f'average duration of nighttime pre-hypo to severe (severe only): {str(data_night.iloc[0, data_night.columns.get_loc("duration_pre_hypo_to_severe")])} +/- {str(data_night.iloc[0, data_night.columns.get_loc("std_pre_hypo_to_severe")])}')
    print(f'median duration of nighttime pre-hypo to severe (severe only): {str(data_night.iloc[0, data_night.columns.get_loc("med_pre_hypo_to_severe")])}')
    print(f'iqr duration of nighttime pre-hypo to severe (severe only): {str(data_night.iloc[0, data_night.columns.get_loc("iqr_pre_hypo_to_severe")])}' + '\n')

    # print rate of hypo onset
    print(f'average rate of onset into hypo (mild only): {str(data_day.iloc[0,data_day.columns.get_loc("avg_rate_mild_only")])} +/- {str(data_day.iloc[0, data_day.columns.get_loc("std_rate_mild_only")])}')
    print(f'median rate of onset into hypo (mild only): {str(data_day.iloc[0,data_day.columns.get_loc("med_rate_mild_only")])}')
    print(f'iqr rate of onset into hypo (mild only): {str(data_day.iloc[0,data_day.columns.get_loc("iqr_rate_mild_only")])}')
    print(f'average rate of onset into hypo (severe only): {str(data_day.iloc[0,data_day.columns.get_loc("avg_rate_severe")])} +/- {str(data_day.iloc[0, data_day.columns.get_loc("std_rate_severe")])}')
    print(f'median rate of onset into hypo (severe only): {str(data_day.iloc[0,data_day.columns.get_loc("med_rate_mild_only")])}')
    print(f'iqr rate of onset into hypo (severe only): {str(data_day.iloc[0,data_day.columns.get_loc("iqr_rate_mild_only")])}')
    print(f'average rate of onset into hypo (mild only): {str(data_night.iloc[0,data_night.columns.get_loc("avg_rate_mild_only")])} +/- {str(data_night.iloc[0, data_night.columns.get_loc("std_rate_mild_only")])}')
    print(f'median rate of onset into hypo (mild only): {str(data_night.iloc[0,data_night.columns.get_loc("med_rate_mild_only")])}')
    print(f'iqr rate of onset into hypo (mild only): {str(data_night.iloc[0,data_night.columns.get_loc("iqr_rate_mild_only")])}')
    print(f'average rate of onset into hypo (severe only): {str(data_night.iloc[0,data_night.columns.get_loc("avg_rate_severe")])} +/- {str(data_night.iloc[0, data_night.columns.get_loc("std_rate_severe")])}')
    print(f'median rate of onset into hypo (severe only): {str(data_night.iloc[0,data_night.columns.get_loc("med_rate_mild_only")])}')
    print(f'iqr rate of onset into hypo (severe only): {str(data_night.iloc[0,data_night.columns.get_loc("iqr_rate_mild_only")])}' + '\n')

    # print min bgl
    print(f'average min BGL for daytime (mild only): {str(data_day.iloc[0,data_day.columns.get_loc("avg_minBGL_mild_only")])} +/- {str(data_day.iloc[0, data_day.columns.get_loc("std_minBGL_mild_only")])}')
    print(f'median min BGL for daytime (mild only): {str(data_day.iloc[0,data_day.columns.get_loc("med_minBGL_mild_only")])}')
    print(f'iqr min BGL for daytime (mild only): {str(data_day.iloc[0,data_day.columns.get_loc("iqr_minBGL_mild_only")])}')
    print(f'average min BGL for daytime (severe only): {str(data_day.iloc[0,data_day.columns.get_loc("avg_minBGL_severe")])} +/- {str(data_day.iloc[0, data_day.columns.get_loc("std_minBGL_severe")])}')
    print(f'median min BGL for daytime (severe only): {str(data_day.iloc[0,data_day.columns.get_loc("med_minBGL_mild_only")])}')
    print(f'iqr min BGL for daytime (severe only): {str(data_day.iloc[0,data_day.columns.get_loc("iqr_minBGL_mild_only")])}')
    print(f'average min BGL for nighttime (mild only): {str(data_night.iloc[0,data_night.columns.get_loc("avg_minBGL_mild_only")])} +/- {str(data_night.iloc[0, data_night.columns.get_loc("std_minBGL_mild_only")])}')
    print(f'median min BGL for nighttime (mild only): {str(data_night.iloc[0,data_night.columns.get_loc("med_minBGL_mild_only")])}')
    print(f'iqr min BGL for nighttime (mild only): {str(data_night.iloc[0,data_night.columns.get_loc("iqr_minBGL_mild_only")])}')
    print(f'average min BGL for nighttime (mild only): {str(data_night.iloc[0,data_night.columns.get_loc("avg_minBGL_severe")])} +/- {str(data_night.iloc[0, data_night.columns.get_loc("std_minBGL_severe")])}')
    print(f'median min BGL for nighttime (severe only): {str(data_night.iloc[0,data_night.columns.get_loc("med_minBGL_mild_only")])}')
    print(f'iqr min BGL for nighttime (severe only): {str(data_night.iloc[0,data_night.columns.get_loc("iqr_minBGL_mild_only")])}' + '\n')

def calcStats(mild, severe, alldata_durations, alldata_num_participants, alldata_patientyear, flag):

    print(f'Number of complete mild events: {len(mild)}')
    print(f'Number of complete severe events: {len(severe)}')

    mild["complete_events"] = ((mild["new_count"] == 0))
    severe["complete_events"] = ((severe["new_count"] == 0))
    mild["parent_events"] = ((mild["new_count"] == 0) & (mild["numSubeventsInParent"] == 1))
    severe["parent_events"] = ((severe["new_count"] == 0) & (severe["numSubeventsInParent"] == 1))
    mild["initial_subevents"] = ((mild["new_count"] >= 1) & (mild["numSubeventsInParent"] >= 2))
    severe["initial_subevents"] = ((severe["new_count"] >= 1) & (severe["numSubeventsInParent"] >= 2))


    print(f'Number of parent mild events: {mild["complete_events"].sum()}')
    print(f'Number of parent severe events: {severe["complete_events"].sum()}')
    # print(f'Number of mild follow-on events == 0: { mild["parent_events"].sum() }')
    # print(f'Number of severe follow-on events == 0: {severe["parent_events"].sum()}')
    print(f'Number of follow-on subevent events (mild) >= 1: {mild["initial_subevents"].sum()}')
    print(f'Number of follow-on subevent events (severe) >= 1: {severe["initial_subevents"].sum()}' + '\n')

    if flag == 0:
        print(f'average number of events per patient hour during the day (mild): {(len(pd.unique(mild["event_id"])) / (alldata_durations / timedelta(hours=1) * (17/24)))}')
        print(f'average number of events per patient hour during the day (severe): {(len(pd.unique(severe["event_id"])) / (alldata_durations / timedelta(hours=1) * (17/24)))}')
    if flag == 1:
        print(f'average number of events per patient hour at night (mild): {(len(pd.unique(mild["event_id"])) / (alldata_durations / timedelta(hours=1) * (7/24)))}')
        print(f'average number of events per patient hour at night (severe): {(len(pd.unique(severe["event_id"])) / (alldata_durations / timedelta(hours=1) * (7/24)))}')

    averages = {}
    # ''' ~~~~~~~~~ DURATIONS ~~~~~~~~~ '''
    severe_only = severe[severe['duration_pre_hypo_to_severe'].notna()].copy()
    severe_treatment = severe[severe['treatment_20_0'].notna()].copy()

    # print(severe_only[["event_id", "duration_mild_to_severe"]])
    averages['duration_severe'] = severe.groupby('event_id')['duration_severe'].sum().round().apply(pd.to_timedelta, unit='s').mean().round('1s')
    averages['std_severe'] = severe.groupby('event_id')['duration_severe'].sum().round().apply(pd.to_timedelta, unit='s').std().round('1s')
    averages['med_severe'] = severe.groupby('event_id')['duration_severe'].sum().round().apply(pd.to_timedelta, unit='s').median().round('1s')
    averages['iqr_severe'] = severe.groupby('event_id')['duration_severe'].sum().round().apply(pd.to_timedelta, unit='s').reset_index().agg(iqr)['duration_severe']

    # prehypo of first child
    averages['duration_prehypo_severe'] = severe.groupby('event_id')['duration_prehypo_to_end'].first().round().apply(pd.to_timedelta, unit='s').mean().round('1s')
    averages['std_prehypo_severe'] = severe.groupby('event_id')['duration_prehypo_to_end'].first().round().apply(pd.to_timedelta, unit='s').std().round('1s')
    averages['med_prehypo_severe'] = severe.groupby('event_id')['duration_prehypo_to_end'].first().round().apply(pd.to_timedelta, unit='s').median().round('1s')
    averages['iqr_prehypo_severe'] = severe.groupby('event_id')['duration_prehypo_to_end'].first().round().apply(pd.to_timedelta, unit='s').reset_index().agg(iqr)['duration_prehypo_to_end']
    averages['duration_prehypo_to_hypo_mean_severe'] = severe.groupby('event_id')['duration_prehypo_to_hypo'].first().round().apply(pd.to_timedelta, unit='s').mean().round('1s')
    averages['std_prehypo_to_hypo_mean_severe'] = severe.groupby('event_id')['duration_prehypo_to_hypo'].first().round().apply(pd.to_timedelta, unit='s').std().round('1s')
    averages['med_prehypo_to_hypo_mean_severe'] = severe.groupby('event_id')['duration_prehypo_to_hypo'].first().round().apply(pd.to_timedelta, unit='s').median().round('1s')
    averages['iqr_prehypo_to_hypo_mean_severe'] = severe.groupby('event_id')['duration_prehypo_to_hypo'].first().round().apply(pd.to_timedelta, unit='s').reset_index().agg(iqr)['duration_prehypo_to_hypo']
    
    # double check that this is correct / should be treatment 20 / get last treatment of the last last child / new element: capture first treatment time in parent event
    averages['duration_treatment_short_severe'] = (severe_treatment.groupby('event_id')['initial_rise_0'].last() - severe_treatment.groupby('event_id')['start_time_hypo'].first()).apply(pd.to_timedelta, unit='s').mean().round('1s')
    averages['std_treatment_short_severe'] = (severe_treatment.groupby('event_id')['initial_rise_0'].last() - severe_treatment.groupby('event_id')['start_time_hypo'].first()).apply(pd.to_timedelta, unit='s').std().round('1s')
    averages['med_treatment_short_severe'] = (severe_treatment.groupby('event_id')['initial_rise_0'].last() - severe_treatment.groupby('event_id')['start_time_hypo'].first()).apply(pd.to_timedelta, unit='s').median().round('1s')
    averages['iqr_treatment_short_severe'] = (severe_treatment.groupby('event_id')['initial_rise_0'].last() - severe_treatment.groupby('event_id')['start_time_hypo'].first()).apply(pd.to_timedelta, unit='s').reset_index().agg(iqr)[0]
    averages['duration_treatment_short_first_severe'] = (severe_treatment.groupby('event_id')['initial_rise_0'].first() - severe_treatment.groupby('event_id')['start_time_hypo'].first()).apply(pd.to_timedelta, unit='s').mean().round('1s')
    averages['std_treatment_short_first_severe'] = (severe_treatment.groupby('event_id')['initial_rise_0'].first() - severe_treatment.groupby('event_id')['start_time_hypo'].first()).apply(pd.to_timedelta, unit='s').std().round('1s')
    averages['med_treatment_short_first_severe'] = (severe_treatment.groupby('event_id')['initial_rise_0'].first() - severe_treatment.groupby('event_id')['start_time_hypo'].first()).apply(pd.to_timedelta, unit='s').median().round('1s')
    averages['iqr_treatment_short_first_severe'] = (severe_treatment.groupby('event_id')['initial_rise_0'].first() - severe_treatment.groupby('event_id')['start_time_hypo'].first()).apply(pd.to_timedelta, unit='s').reset_index().agg(iqr)[0]
    averages['avg_minBGL_severe'] = severe.groupby('event_id')['minBGL'].min().round(1).mean().round(1)
    averages['std_minBGL_severe'] = severe.groupby('event_id')['minBGL'].min().round(1).std().round(1)
    averages['med_minBGL_severe'] = severe.groupby('event_id')['minBGL'].min().round(1).median().round(1)
    averages['iqr_minBGL_severe'] = severe.groupby('event_id')['minBGL'].min().round(1).reset_index().agg(iqr).round(1)["minBGL"]

    # gets the first row of a group
    averages['avg_rate_severe'] = severe.groupby('event_id')['rate_onset_into_hypo'].first().mean().round(2)
    averages['std_rate_severe'] = severe.groupby('event_id')['rate_onset_into_hypo'].first().std().round(2)
    averages['med_rate_severe'] = severe.groupby('event_id')['rate_onset_into_hypo'].first().median().round(2)
    averages['iqr_rate_severe'] = severe.groupby('event_id')['rate_onset_into_hypo'].first().reset_index().agg(iqr).round(1)['rate_onset_into_hypo']

    # take first value / duration of mild before severe (window of opportunity) / we need both sum and first
    averages['duration_mild_to_severe_first'] = severe_only.groupby('event_id')['duration_mild_to_severe'].first().round().apply(pd.to_timedelta, unit='s').mean().round('1s')
    averages['std_mild_to_severe_first'] = severe_only.groupby('event_id')['duration_mild_to_severe'].first().round().apply(pd.to_timedelta, unit='s').std().round('1s')
    averages['med_mild_to_severe_first'] = severe_only.groupby('event_id')['duration_mild_to_severe'].first().round().apply(pd.to_timedelta, unit='s').median().round('1s')
    averages['iqr_mild_to_severe_first'] = severe_only.groupby('event_id')['duration_mild_to_severe'].first().round().apply(pd.to_timedelta, unit='s').reset_index().agg(iqr)['duration_mild_to_severe']
    averages['duration_mild_to_severe'] = severe_only.groupby('event_id')['duration_mild_to_severe'].sum().round().apply(pd.to_timedelta, unit='s').mean().round('1s')
    averages['std_mild_to_severe'] = severe_only.groupby('event_id')['duration_mild_to_severe'].sum().round().apply(pd.to_timedelta, unit='s').std().round('1s')
    averages['med_mild_to_severe'] = severe_only.groupby('event_id')['duration_mild_to_severe'].sum().round().apply(pd.to_timedelta, unit='s').median().round('1s')
    averages['iqr_mild_to_severe'] = severe_only.groupby('event_id')['duration_mild_to_severe'].sum().round().apply(pd.to_timedelta, unit='s').reset_index().agg(iqr)['duration_mild_to_severe']
    averages['duration_pre_hypo_to_severe'] = severe_only.groupby('event_id')['duration_pre_hypo_to_severe'].first().round().apply(pd.to_timedelta, unit='s').mean().round('1s')
    averages['std_pre_hypo_to_severe'] = severe_only.groupby('event_id')['duration_pre_hypo_to_severe'].first().round().apply(pd.to_timedelta, unit='s').std().round('1s')
    averages['med_pre_hypo_to_severe'] = severe_only.groupby('event_id')['duration_pre_hypo_to_severe'].first().round().apply(pd.to_timedelta, unit='s').median().round('1s')
    averages['iqr_pre_hypo_to_severe'] = severe_only.groupby('event_id')['duration_pre_hypo_to_severe'].first().round().apply(pd.to_timedelta, unit='s').reset_index().agg(iqr)['duration_pre_hypo_to_severe']

    averages['duration_sum_severe'] = severe.groupby('event_id')['duration_severe'].sum().round().apply(pd.to_timedelta, unit='s').sum().round('1s')

    # hypo exclusive (no severe)
    averages['duration_hypo_mild_only'] = mild.groupby('event_id')['duration_hypo'].sum().round().apply(pd.to_timedelta, unit='s').mean().round('1s')
    averages['std_hypo_mild_only'] = mild.groupby('event_id')['duration_hypo'].sum().round().apply(pd.to_timedelta, unit='s').std().round('1s')
    averages['med_hypo_mild_only'] = mild.groupby('event_id')['duration_hypo'].sum().round().apply(pd.to_timedelta, unit='s').median().round('1s')
    averages['iqr_hypo_mild_only'] = mild.groupby('event_id')['duration_hypo'].first().round().apply(pd.to_timedelta, unit='s').reset_index().agg(iqr)['duration_hypo']
    averages['duration_prehypo_mild_only'] = mild.groupby('event_id')['duration_prehypo_to_end'].first().round().apply(pd.to_timedelta, unit='s').mean().round('1s')
    averages['std_prehypo_mild_only'] = mild.groupby('event_id')['duration_prehypo_to_end'].first().round().apply(pd.to_timedelta, unit='s').std().round('1s')
    averages['med_prehypo_mild_only'] = mild.groupby('event_id')['duration_prehypo_to_end'].first().round().apply(pd.to_timedelta, unit='s').median().round('1s')
    averages['iqr_prehypo_mild_only'] = mild.groupby('event_id')['duration_prehypo_to_end'].first().round().apply(pd.to_timedelta, unit='s').reset_index().agg(iqr)['duration_prehypo_to_end']
    averages['duration_prehypo_to_hypo_mild_only'] = mild.groupby('event_id')['duration_prehypo_to_hypo'].first().round().apply(pd.to_timedelta, unit='s').mean().round('1s')
    averages['std_prehypo_to_hypo_mild_only'] = mild.groupby('event_id')['duration_prehypo_to_hypo'].first().round().apply(pd.to_timedelta, unit='s').std().round('1s')
    averages['med_prehypo_to_hypo_mild_only'] = mild.groupby('event_id')['duration_prehypo_to_hypo'].first().round().apply(pd.to_timedelta, unit='s').median().round('1s')
    averages['iqr_prehypo_to_hypo_mild_only'] = mild.groupby('event_id')['duration_prehypo_to_hypo'].first().round().apply(pd.to_timedelta, unit='s').reset_index().agg(iqr)['duration_prehypo_to_hypo']
    averages['avg_minBGL_mild_only'] = mild.groupby('event_id')['minBGL'].min().round().mean().round(1)
    averages['std_minBGL_mild_only'] = mild.groupby('event_id')['minBGL'].min().round().std().round(1)
    averages['med_minBGL_mild_only'] = mild.groupby('event_id')['minBGL'].min().round().median().round(1)
    averages['iqr_minBGL_mild_only'] = mild.groupby('event_id')['minBGL'].min().round().reset_index().agg(iqr).round(1)['minBGL']
    
    # need to double check calculations of rate
    averages['avg_rate_mild_only'] = mild.groupby('event_id')['rate_onset_into_hypo'].first().mean().round(2)
    averages['std_rate_mild_only'] = mild.groupby('event_id')['rate_onset_into_hypo'].first().std().round(2)
    averages['med_rate_mild_only'] = mild.groupby('event_id')['rate_onset_into_hypo'].first().median().round(2)
    averages['iqr_rate_mild_only'] = mild.groupby('event_id')['rate_onset_into_hypo'].first().reset_index().agg(iqr).round(2)['rate_onset_into_hypo']
    
    averages['duration_sum_mild'] = mild.groupby('event_id')['duration_hypo'].sum().round().apply(pd.to_timedelta, unit='s').sum().round('1s')

    return averages


def calcStatsR(mild, severe, alldata_durations, alldata_num_participants, flag):
    
    print(f'Number of mild events: {len(mild)}')
    print(f'Number of severe events: {len(severe)}')

    mild["parent_events"] = ((mild["new_count"] == 0) & (mild["numSubeventsInParent"] == 1))
    severe["parent_events"] = ((severe["new_count"] == 0) & (severe["numSubeventsInParent"] == 1))
    mild["initial_subevents"] = ((mild["new_count"] == 1) & (mild["numSubeventsInParent"] == 2))
    severe["initial_subevents"] = ((severe["new_count"] == 1) & (severe["numSubeventsInParent"] == 2))
    mild["followon_subevents"] = ((mild["new_count"] >= 2) & (mild["numSubeventsInParent"] >= 3))
    severe["followon_subevents"] = ((severe["new_count"] >= 2) & (severe["numSubeventsInParent"] >= 3))

    if flag == 0:
        print(f'average number of events per patient hour during the day (mild): {(len(pd.unique(mild["event_id"])) / (alldata_durations / timedelta(hours=1) * (17/24)))}')
        print(f'average number of events per patient hour during the day (severe): {(len(pd.unique(severe["event_id"])) / (alldata_durations / timedelta(hours=1) * (17/24)))}')
    if flag == 1:
        print(f'average number of events per patient hour at night (mild): {(len(pd.unique(mild["event_id"])) / (alldata_durations / timedelta(hours=1) * (7/24)))}')
        print(f'average number of events per patient hour at night (severe): {(len(pd.unique(severe["event_id"])) / (alldata_durations / timedelta(hours=1) * (7/24)))}')

    durations = {}
    # ''' ~~~~~~~~~ DURATIONS ~~~~~~~~~ '''
    severe_only = severe[severe['duration_pre_hypo_to_severe'].notna()].copy()
    severe_treatment = severe[severe['treatment_20_0'].notna()].copy()

    severe['duration_severe'] = severe.groupby('event_id')['duration_severe'].sum().round().apply(pd.to_timedelta, unit='s')
    severe['duration_prehypo_severe'] = severe.groupby('event_id')['duration_prehypo_to_end'].first().round().apply(pd.to_timedelta, unit='s')
    severe['duration_prehypo_to_hypo_mean_severe'] = severe.groupby('event_id')['duration_prehypo_to_hypo'].first().round().apply(pd.to_timedelta, unit='s')
    severe_treatment['duration_treatment_short_severe'] = (severe_treatment.groupby('event_id')['initial_rise_0'].last() - severe_treatment.groupby('event_id')['start_time_hypo'].first()).apply(pd.to_timedelta, unit='s')
    severe_treatment['duration_treatment_short_first_severe'] = (severe_treatment.groupby('event_id')['initial_rise_0'].first() - severe_treatment.groupby('event_id')['start_time_hypo'].first()).apply(pd.to_timedelta, unit='s')
    severe['avg_minBGL_severe'] = severe.groupby('event_id')['minBGL'].min().round(1)
    severe['avg_rate_severe'] = severe.groupby('event_id')['rate_onset_into_hypo'].first()
    severe_only['duration_mild_to_severe_first'] = severe_only.groupby('event_id')['duration_mild_to_severe'].first().round().apply(pd.to_timedelta, unit='s')
    severe_only['duration_mild_to_severe'] = severe_only.groupby('event_id')['duration_mild_to_severe'].sum().round().apply(pd.to_timedelta, unit='s')
    severe_only['duration_pre_hypo_to_severe'] = severe_only.groupby('event_id')['duration_pre_hypo_to_severe'].first().round().apply(pd.to_timedelta, unit='s')

    mild['duration_hypo_mild_only'] = mild.groupby('event_id')['duration_hypo'].sum().round().apply(pd.to_timedelta, unit='s')
    mild['duration_prehypo_mild_only'] = mild.groupby('event_id')['duration_prehypo_to_end'].first().round().apply(pd.to_timedelta, unit='s')
    mild['duration_prehypo_to_hypo_mild_only'] = mild.groupby('event_id')['duration_prehypo_to_hypo'].first().round().apply(pd.to_timedelta, unit='s')
    mild['avg_minBGL_mild_only'] = mild.groupby('event_id')['minBGL'].min().round(1)
    mild['avg_rate_mild_only'] = mild.groupby('event_id')['rate_onset_into_hypo'].first()

    return mild, severe, severe_only, severe_treatment


def splitGroupedEventsIntoMildAndSevereData(df):
    # convert timestamps to datetime
    df.start_time_pre_hypo = pd.to_datetime(df.start_time_pre_hypo, errors='coerce')
    df.start_time_hypo = pd.to_datetime(df.start_time_hypo, errors='coerce')
    df.start_time_severe = pd.to_datetime(df.start_time_severe, errors='coerce')
    df.end_hypo_time = pd.to_datetime(df.end_hypo_time, errors='coerce')
    df.global_min_time = pd.to_datetime(df.global_min_time, errors='coerce')
    df.treatment_20_0 = pd.to_datetime(df.treatment_20_0, errors='coerce')
    df.initial_rise_0 = pd.to_datetime(df.initial_rise_0, errors='coerce')
    df.last_full_rise = pd.to_datetime(df.last_full_rise, errors='coerce')

    # capture time data
    df['pre_hypo_time'] = df.start_time_pre_hypo.dt.time
    df['hypo_time'] = df.start_time_hypo.dt.time
    df['severe_time'] = df.start_time_severe.dt.time

    col_names = ['id', 'minBGL', 'global_min_time', 'start_time_hypo', 'start_time_hypo_gl', 'start_time_pre_hypo', 'start_time_pre_hypo_gl',
       'last_full_rise', 'last_full_rise_gl', 'end_hypo_time', 'end_hypo_time_gl', 'start_time_severe', 'start_time_severe_gl',
       'initial_rise_0', 'initial_rise_gl_0', 'initial_rise_1', 'initial_rise_gl_1', 'treatment_20_0', 'treatment_20_gl_0', 'treatment_20_1', 
       'treatment_20_gl_1', 'new_count', 'event_id', 'tod', 'pre_hypo_time', 'hypo_time', 'severe_time']
    severe = pd.DataFrame(columns=col_names)
    mild = pd.DataFrame(columns=col_names)   

    for event, grouped_event in df.groupby(['event_id']):
        if (min(grouped_event.minBGL)) <= 54:
            severe = severe.append(grouped_event)
        else:
            mild = mild.append(grouped_event)

    # mild.to_csv('../KineticsOfHypoglycemiaAnalysis/data/processed/df_mild.csv')
    # severe.to_csv('../KineticsOfHypoglycemiaAnalysis/data/processed/df_severe.csv')
    return mild, severe
    

def splitAllEventsIntoMildAndSevere(df):
    ''' THIS CREATES DURATION DATASETS AND SPLITS THE DATA BETWEEN MILD AND SEVERE '''
    # convert timestamps to datetime
    df.start_time_pre_hypo = pd.to_datetime(df.start_time_pre_hypo, errors='coerce')
    df.start_time_hypo = pd.to_datetime(df.start_time_hypo, errors='coerce')
    df.start_time_severe = pd.to_datetime(df.start_time_severe, errors='coerce')
    df.end_hypo_time = pd.to_datetime(df.end_hypo_time, errors='coerce')
    df.global_min_time = pd.to_datetime(df.global_min_time, errors='coerce')
    # df.treatment_20_0 = pd.to_datetime(df.treatment_20_0, errors='coerce')
    df.initial_rise_0 = pd.to_datetime(df.initial_rise_0, errors='coerce')
    df.last_full_rise = pd.to_datetime(df.last_full_rise, errors='coerce')

    # # capture time data
    df['pre_hypo_time'] = df.start_time_pre_hypo.dt.time
    df['hypo_time'] = df.start_time_hypo.dt.time
    df['severe_time'] = df.start_time_severe.dt.time

    severe = df.copy(deep=True)
    severe = severe[severe['start_time_severe'].notna()]
    mild = df.copy(deep=True)
    mildCond = mild['severe_time'].isin(severe['severe_time'])
    mild.drop(mild[mildCond].index, inplace=True)

    return mild, severe


def statPrep(mild, severe):
    oneday = pd.Timedelta(minutes=1)

    # calc df level data
    mild['duration_prehypo_to_end'] = (mild.end_hypo_time - mild.start_time_pre_hypo).dt.total_seconds()
    mild['duration_prehypo_to_hypo'] = (mild.start_time_hypo - mild.start_time_pre_hypo).dt.total_seconds()
    mild['duration_hypo'] = (mild.end_hypo_time - mild.start_time_hypo).dt.total_seconds()
    # df['duration_treatment_short'] = df.apply(lambda row: row['treatment_20_0'] if row['treatment_20_0']==pd.NaN else row['initial_rise_0'] - row['start_time_hypo'], axis=1)
    mild['initial_rise_0'] = (np.where(mild['treatment_20_0'].isnull(),mild['treatment_20_0'],mild['initial_rise_0']))
    # mild.initial_rise_0 = pd.to_datetime(mild.initial_rise_0, errors='coerce')

    # mild['duration_treatment_short'] = (mild.initial_rise_0 - mild.start_time_hypo).dt.total_seconds()
    # df['duration_treatment_short'] = np.where(df['treatment_20_0'].isnull(), "NaN", df['duration_treatment_short'])
    mild['duration_treatment_long'] = (mild.end_hypo_time - mild.global_min_time).dt.total_seconds()
    mild['rate_onset_into_hypo'] = (mild.start_time_pre_hypo_gl - mild.start_time_hypo_gl) / ((mild.start_time_pre_hypo - mild.start_time_hypo) / oneday) # rate calcs delta gl / delta time
    inf_replace = mild.loc[mild['rate_onset_into_hypo'] != np.inf, 'rate_onset_into_hypo'].max()
    mild['rate_onset_into_hypo'].replace(np.inf,inf_replace,inplace=True)

    severe['duration_severe'] = (severe.end_hypo_time - severe.start_time_severe).dt.total_seconds()
    severe['duration_prehypo_to_end'] = (severe.end_hypo_time - severe.start_time_pre_hypo).dt.total_seconds()
    severe['duration_prehypo_to_hypo'] = (severe.start_time_hypo - severe.start_time_pre_hypo).dt.total_seconds()
    severe['duration_hypo'] = (severe.end_hypo_time - severe.start_time_hypo).dt.total_seconds()
    severe['initial_rise_0'] = np.where(severe['treatment_20_0'].isnull(),severe['treatment_20_0'],severe['initial_rise_0'])
    severe.initial_rise_0 = pd.to_datetime(severe.initial_rise_0, errors='coerce')

    severe['duration_treatment_short'] = (severe.initial_rise_0 - severe.start_time_hypo).dt.total_seconds()
    severe['duration_treatment_long'] = (severe.end_hypo_time - severe.global_min_time).dt.total_seconds()
    diff_gl = severe.start_time_pre_hypo_gl - severe.start_time_hypo_gl
    diff_time = (severe.start_time_pre_hypo - severe.start_time_hypo) / oneday
    severe['rate_onset_into_hypo'] = diff_gl.div(diff_time.replace(0, np.nan)).fillna(0) # rate calcs delta gl / delta time
    severe['duration_mild_to_severe'] = (severe.start_time_severe - severe.start_time_hypo).dt.total_seconds()
    severe['duration_pre_hypo_to_severe'] = (severe.start_time_severe - severe.start_time_pre_hypo).dt.total_seconds()

    # print(f'The length of mild data: {len(pd.unique(mild["event_id"]))}')
    # print(f'The length of severe data: {len(pd.unique(severe["event_id"]))}')

    print(f'the number of nan values: {mild["initial_rise_0"].isna().sum()}')

    return mild, severe