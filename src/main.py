import pandas as pd
from features import preprocessing, cleanBuckingham, addNewData
from models import analysis
from visualization import plottingData as vs
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import h5py
from tqdm import tqdm
from features import fixTime as ft


if __name__ == '__main__':

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ GLOBAL VARIABLES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    main_folder = "/data/"
    author = "aleppo"
    dataset_raw_folder = f"{main_folder}raw/{author}/{author}_processed.csv"
    dataset_interim_folder = f"{main_folder}interim/individualData/{author}/"
    dataset_acceptable_files = f"{main_folder}interim/individualData/acceptableFiles_{author}_processed.txt"
    dataset_processed_folder = f"{main_folder}processed/cleanedData/"
    dataset_processed_file = f"{main_folder}processed/data_{author}_processed.csv"
    dataset_cleaned_file = f"{dataset_processed_folder}data_{author}_processed_cleaned.csv"


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ PREPROCESSING RAW DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    # PREPROCESSING: parses CGM datasets and outputs the dictionary of events
    # STEP 0: QUESTION EVERYTHING
    # df = preprocessing.genCSVForAnalysis(dataset_raw_folder)
    # df['gl'] = df['gl'].astype(int)
    # s = df['gl'].ge(70).mean()
    # print(s)
    # single = df[df.id == '1_1']
    # print(single)
    # ax = single.set_index('time', inplace=True)
    # ax = single['gl'].plot(legend=True)
    # ax.set_ylim([30, 100])
    # df.groupby('id')['gl'].plot(legend=True)
    # plt.show()

    # PLOTS OUT acceptable files for 1_1
    # with open(dataset_acceptable_files,'r') as fd:
    #     all_csv_files = [filename.rstrip() for filename in fd.readlines()]
    # for file in all_csv_files:
    #     if '1_1_' in file: 
    #         df = pd.read_csv(file, sep=",")
    #         ax = df.set_index('time', inplace=True)
    #         ax = df['gl'].plot(legend=True)
    #         ax.set_ylim([30, 100])
    #         plt.show()


    # STEP 1
    # read in output of importRawData.py
    # df = preprocessing.genCSVForAnalysis(dataset_raw_folder)
    # # segment dataset by id and timegaps larger than 60 min
    # preprocessing.cutFiles(df, dataset_interim_folder)

    # # # STEP 2
    # extention = "*.csv"
    # all_csv_files = [file
    #                  for path, subdir, files in os.walk(dataset_interim_folder)
    #                  for file in glob(os.path.join(dataset_interim_folder, extention))]

    # # STEP 2.A
    # with open(dataset_acceptable_files, 'w') as fd:
    #     for file in tqdm(all_csv_files):
    #         df = pd.read_csv(file, sep=",")
    #         # df.gl = df.gl.apply(str)
    #         # df.gl = df.gl.mask(df.gl == 'Low', '40')
    #         # df.gl = df.gl.apply(int)
    #         if min(df.gl) > 69 or df.shape[0] < 7:
    #             pass
    #             # print(f'not saving {file}')
    #         else:
    #             fd.write(f'{file}\n')

    # # STEP 2B: initial preprocessing
    # with open(dataset_acceptable_files,'r') as fd:
    #     all_csv_files = [filename.rstrip() for filename in fd.readlines()]
    # for file in tqdm(all_csv_files):
    #     df = pd.read_csv(file, sep=",")
    #     df.time = pd.to_datetime(df.time, errors='coerce')
    #     df = df.set_index('time', drop=True)
    #     # df.gl = df.gl.apply(str)
    #     # df.gl = df.gl.mask(df.gl == 'Low', '40')
    #     # df.gl = df.gl.apply(int)
    #     preprocessing.parse(df, dataset_processed_file)

    # # STEP 3: set up for calculating the full durations of the data
    # csv = f"{main_folder}processed/fulldata_{author}.csv"
    # with open(dataset_acceptable_files,'r') as fd:
    #     all_csv_files = [filename.rstrip() for filename in fd.readlines()]
    # for file in tqdm(all_csv_files):
    #     df = pd.read_csv(file, sep=",")
    #     df.time = pd.to_datetime(df.time, errors='coerce')
    #     df = df.set_index('time', drop=True)
    #     df.to_csv(csv, mode='a', index=False, header=False)

    # # STEP 4: cleaning the datasets
    # preprocessing.cleanDataset(dataset_processed_file, dataset_processed_folder)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ ANALYSIS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    # PART A: DURATIONS: CALCULATING DURATIONS OF ALL DATA (not snippets)

    # STEP 1: Create new dataframes that output the start and end times of each file
    # all_duration_data = []

    # extention = "*.csv"
    # dataset_acceptable_files = [file
    #                  for path, subdir, files in os.walk(dataset_interim_folder)
    #                  for file in glob(os.path.join(dataset_interim_folder, extention))]
    
    # all_duration_event = {}
    # for file in tqdm(dataset_acceptable_files):
    #     df = pd.read_csv(file, sep=",")
    #     all_duration_event['studyid'] = df.id.iloc[0]
    #     all_duration_event['data_num'] = df.dataCount.iloc[0]
    #     all_duration_event['start_time'] = df.time.iloc[0]
    #     all_duration_event['end_time'] = df.time.iloc[-1]
    #     all_duration_data.append(all_duration_event)
    #     all_duration_event = {}

    # df_duration_data = pd.DataFrame(all_duration_data)
    # df_duration_data.to_csv(f'{main_folder}interim/individualData/durations/{author}_processed_all_files_durations.csv', sep=',', encoding='utf-8')

    # STEP 2: CALCULATE DURATIONS OF ALL DATA
    # preprocessing.calcDurations(f'{main_folder}interim/individualData/durations/{author}_processed_all_files_durations.csv')
    # print("completed calculating durations")

    general_population = ["chase", "tamborlane", "weinstock", "buckingham"]
    # general_population = ["chase", "tamborlane", "weinstock", "buckingham", "aleppo"]
    # # # PART B: CALC STATS ON DATA
    df_day, df_night, alldata_durations, alldata_num_participants, alldata_patientyear = analysis.prepFilesForStats(general_population)
    analysis.getStats(df_day, df_night, alldata_durations, alldata_num_participants, alldata_patientyear, author)
    # analysis.parseStats(data, alldata_durations, alldata_num_participants)
    # ft.addTodToCleanedData(dataset`_cleaned_file, author)

    # PART Ca: PREPARE FOR VISUALIZATION
    # preprocessing.processForVis(dataset_cleaned_file, dataset_interim_folder, dataset_acceptable_files)


    # # PART Cb: 
    # file = f'{main_folder}processed/visData/visualizationData_{author}.csv'
    # df = pd.read_csv(file, sep=",")
    # # df["count"] = '0'

    # # datacount = 0
    # # prevDatacount = 0
    # # wasPrevAHeader = False
    # # for index, row in tqdm(df.iterrows()):
    # #     # CHASE AND BUCKINGHAM ARE COLUMN 3 AND THE REST ARE COLUMN 2
    # #     if 'id' in row[2] and wasPrevAHeader:
    # #         wasPrevAHeader = True
    # #         pass
    # #     elif 'id' in row[2] and not wasPrevAHeader:
    # #         datacount += 1
    # #         df.at[index, 'count'] = str(datacount)
    # #         prevDatacount = datacount
    # #         wasPrevAHeader = True
    # #     else:
    # #         df.at[index, 'count'] = str(prevDatacount)
    # #         wasPrevAHeader = False

    # df = df[df.id.str.contains('id') == False]
    # df = df[df['time'].notna()]
    # df['ages'] = '<18'
    # # df['count'] = pd.to_numeric(df['count'])
    # print(df)


    # # convert timestamps to datetime
    # df.time = pd.to_datetime(df.time, errors='coerce')
    # df = df.set_index('time', drop=False)
    # df = df.rename(columns={"Unnamed: 0": "order", "time": "datetime"})
    # # for chase, buckingham
    # df = df.drop(columns={'Unnamed: 0.2', 'Unnamed: 0.1'})
    # # for aleppo, weinstock, tamborlane
    # # df = df.drop(columns={'Unnamed: 0.1'})
    # # mask_day = df.between_time('06:00:00', '23:00:00', include_start=True, include_end=False)
    # # mask_day['timeOfDay'] = "Diurnal"
    # # mask_night = df.drop(df.between_time('06:00:00', '23:00:00', include_start=True, include_end=False))
    # # mask_night['timeOfDay'] = "Nocturnal"
    # # df_all = pd.concat([mask_night, mask_day])
    # # df_all = df.sort_values(by = ['event_id'])
    # # print(df_all)
    # df.to_csv(f"{main_folder}processed/visData/visualizationData_{author}_withmetadata.csv")

    # VISUALIZATION OF ANALYSIS COUNTS
    # visdata = pd.read_csv(f'{main_folder}processed/visData/eventCounts.csv', sep=',', encoding='utf-8')
    # vs.plotSubEventNumbers(visdata)
    # vs.plotInitialVsFollowOn(visdata)
    # do percentage and absolute numbers
    # vs.occurrenceOfHypo(visdata)