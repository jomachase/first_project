import pandas as pd

def clean_gender(gender: str) -> str:
	if gender in ["M", "F"]:
		return gender
	elif gender in ["m", "male", "Male"]:
		return "M"
	elif gender in ["f", "female", "Female"]:
		return "F"
	else:
		return "Unknown"

def clean_colum_names(df: pd.DataFrame) -> pd.DataFrame:
	df2 = df.copy()
	
	df2.columns = [ col.replace(" ","_") for col in df2.columns ]

	return df2


