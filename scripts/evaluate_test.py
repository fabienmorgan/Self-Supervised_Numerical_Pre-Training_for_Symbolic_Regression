import pandas as pd

filename = "evaluation_of_prediction_mmsr_train_nc_SupportRange-100to100.csv"

equation_prediction = pd.read_csv(filename)

total_entries = len(equation_prediction)

match_sum = equation_prediction['Match Equation'].sum()

match_percentage = (match_sum / total_entries) * 100

print(f'Total Entries: {total_entries}')

print(f'Match Sum: {match_sum}')

print(f'Match Percentage: {match_percentage}%')