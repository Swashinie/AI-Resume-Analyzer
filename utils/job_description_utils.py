# utils/job_description_utils.py

import pandas as pd

def load_and_preprocess_job_descriptions(csv_path):
    """
    Load job descriptions from a CSV file and preprocess the text.

    Args:
        csv_path (str): Path to the CSV file containing job descriptions.

    Returns:
        pd.DataFrame: DataFrame with original and cleaned job descriptions.
    """
    job_desc_df = pd.read_csv(csv_path)

    def clean_text(text):
        if isinstance(text, str):
            return text.lower().strip()
        return ""

    # Use the correct column name from your CSV file
    job_desc_df['cleaned_description'] = job_desc_df['Resume_str'].apply(clean_text)
    return job_desc_df


if __name__ == "__main__":
    # Example usage
    job_desc_path = 'data/job_descriptions/job_descriptions.csv'
    job_descriptions = load_and_preprocess_job_descriptions(job_desc_path)
    print(job_descriptions.head())
