import pandas as pd
import os

def datacleaning(df) :
    try :
        if not os.path.exists(df) :
            raise FileNotFoundError(f"Dataset file not found: {df}")
        
        clean_data = pd.read_csv(df)
        
        if clean_data.empty :
            raise ValueError("Dataset is empty")
        
        clean_data = clean_data.dropna()
        
        if clean_data.empty :
            raise ValueError("Dataset is empty after removing missing values")
        
        return clean_data
    
    except FileNotFoundError as e :
        raise e
    except pd.errors.EmptyDataError :
        raise ValueError("CSV file is empty or corrupted")
    except Exception as e :
        raise Exception(f"Error during data cleaning: {str(e)}")