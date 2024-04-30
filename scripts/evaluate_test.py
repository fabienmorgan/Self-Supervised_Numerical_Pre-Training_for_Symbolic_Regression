import os
import json
import pandas as pd

path = "evaluation/"
summary_filename = 'evaluation_summary.csv'
evaluation_data = []

for filename in os.listdir(path):
    if filename.endswith(".json"):
        with open(os.path.join(path, filename), 'r') as json_file:
            data = json.load(json_file)

            csv_filename = data['csv_filename']

            equation_prediction = pd.read_csv(os.path.join(path, csv_filename))
            
            total_entries = len(equation_prediction)
            match_sum = equation_prediction['Match Equation'].sum()
            match_percentage = (match_sum / total_entries) * 100

            evaluation_data.append({
                'Model': data['model'],
                'Minimum Support': data['minimum_support'],
                'Maximum Support': data['maximum_support'],
                'Total Entries': total_entries,
                'Match Sum': match_sum,
                'Match Percentage': match_percentage
            })
            
evaluation_df = pd.DataFrame(evaluation_data)

evaluation_df.to_csv(os.path.join(path,summary_filename), index=False)           