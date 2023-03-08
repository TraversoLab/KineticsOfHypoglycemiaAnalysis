import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings


def fixTime(df):
	# print(df)
	df["newTime"] = 0.0
	df.rename(columns={"id.x": "id"}, inplace=True)
	df.phaseLabel = df.phaseLabel.astype(str)
	for name, group in tqdm(df.groupby(["event_id"])):
		# print(group)
		flag = 0
		prevValue = 0.0
		for index, row in group.iterrows():
			# print(row.phaseLabel)
			if "pre" not in row.phaseLabel and flag == 0:
				df.loc[index, "newTime"] = 0.0
				prevValue = 5.0 
				flag = 1
			elif flag == 1:
				df.loc[index, "newTime"] = prevValue
				prevValue += 5.0
			else:
				pass
	df = df[::-1]
	# print(df)

	for name, group in tqdm(df.groupby(["event_id"])):
		flag_r = 0
		prevValue_r = 0.0
		for index, row in group.iterrows():
			if "pre" in row.phaseLabel and flag_r == 0:
				df.loc[index, "newTime"] = -5.0
				prevValue_r = -10
				flag_r = 1
			elif "pre" in row.phaseLabel and flag_r == 1:
				df.loc[index, "newTime"] = prevValue_r
				prevValue_r += -5.0
			else:
				flag_r = 0
	df = df[::-1]
	# print(df)
	return df


def remakeToD(df):
	df.start_time_hypo = pd.to_datetime(df.start_time_hypo)
	df = df.set_index('start_time_hypo', drop=False)
	mask_day = df.between_time('06:00:00', '23:00:00')
	mask_day['timeOfDay'] = "Diurnal"
	print(len(mask_day))
	mask_night = df.drop(df.between_time('06:00', '23:00').index)
	mask_night['timeOfDay'] = "Nocturnal"
	print(len(mask_night))
	df_all = pd.concat([mask_night, mask_day])
	df_all['isTreatedData'] = np.where(df.id.str.contains("6_"), "Treated", "Untreated")

	# df_all = df_all.sort_values(by = ['count'])
	return df_all

def addTodToCleanedData(df, author):
	warnings.filterwarnings("ignore")
	df = pd.read_csv(df)
	df.drop(columns="Unnamed: 0", inplace=True)

	df.start_time_hypo = pd.to_datetime(df.start_time_hypo, errors='coerce')
	df = df.set_index('start_time_hypo', drop=False)
	df['tod'] = df.index.isin(df.between_time('23:00:00', '06:00:00', include_start=True, include_end=False).index)
	df['tod'] = df['tod'].astype(int) # false = 0 (day); true = 1 (night)

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

	df['numSubeventsInParent'] = df.groupby('event_id')['new_count'].transform('size')
	df = df.reset_index(drop=True)
	for event, grouped_event in df.groupby(['event_id']):
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
	
	df_day['timeOfDay'] = "Diurnal"
	df_night['timeOfDay'] = "Nocturnal"
	df = pd.concat([df_night, df_day])
	# print(df)
	df.to_csv(f"../KineticsOfHypoglycemiaAnalysis/data/processed/cleanedData/data_{author}_cleaned.csv")

if __name__ == '__main__':
	file = "../KineticsOfHypoglycemiaAnalysis/data/processed/aleppo.csv"
	df = pd.read_csv(file, sep=",")
	data = fixTime(df)
	data.to_csv("../KineticsOfHypoglycemiaAnalysis/data/processed/timedData_aleppo.csv")