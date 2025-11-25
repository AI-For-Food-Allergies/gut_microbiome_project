import pandas as pd
import os

def preprocess_gadir(input_csv: str, output_dir: str = None):
    """
    Preprocess the Gadir metadata CSV file and create separate CSV files for each age group.
    
    Parameters:
    -----------
    input_csv : str
        Path to the input CSV file (gadir_metadata.csv)
    output_dir : str, optional
        Directory to save output CSV files. If None, defaults to the same directory as input_csv.
    
    Returns:
    --------
    dict
        Dictionary mapping age_group names to their corresponding DataFrames
    """
    # Load the dataset
    df = pd.read_csv(input_csv)
    
    # Filter for human samples only (exclude any non-human samples like mice)
    # Check both HOST and Organism columns to ensure we only have human samples
    if 'HOST' in df.columns:
        df = df[df['HOST'].str.contains('Homo sapiens', case=False, na=False)]
    if 'Organism' in df.columns:
        df = df[df['Organism'].str.contains('human', case=False, na=False)]
    
    # Filter out Unclear/unclear samples (case-insensitive)
    df = df[~df['Group'].str.lower().isin(['unclear'])]
    
    # Map Group values to binary labels
    label_mapping = {
        'FoodAllergy': 1,
        'Control': 0,
        'ControlHiRisk': 0
    }
    
    df['label'] = df['Group'].map(label_mapping)
    
    # Create age groups based on 4-6 month sampling intervals
    # Study samples infants every 4-6 months for up to 30 months
    def assign_age_group(age):
        """Assign age group based on 6-month sampling intervals."""
        if pd.isna(age):
            return None
        age = float(age)
        if age < 6:
            return '0-6_months'
        elif age < 12:
            return '6-12_months'
        elif age < 18:
            return '12-18_months'
        elif age < 24:
            return '18-24_months'
        elif age < 30:
            return '24-30_months'
        else:
            return '30+_months'
    
    # Add age group column if Age_at_Collection exists
    if 'Age_at_Collection' not in df.columns:
        raise ValueError("Age_at_Collection column not found in the input CSV file.")
    
    df['age_group'] = df['Age_at_Collection'].apply(assign_age_group)
    
    # Ensure proper data types
    df['Run'] = df['Run'].astype(str)
    df['label'] = df['label'].astype(int)
    
    # Set output directory if not provided
    if output_dir is None:
        output_dir = os.path.dirname(input_csv)
    
    # Group by age_group and create separate CSV files
    age_groups = df['age_group'].dropna().unique()
    grouped_dataframes = {}
    
    for age_group in sorted(age_groups):
        # Filter data for this age group
        df_group = df[df['age_group'] == age_group][['Run', 'label']].copy()
        
        # Create output filename
        output_filename = f'gadir_preprocessed_{age_group}.csv'
        output_path = os.path.join(output_dir, "downloaded_data", output_filename)
        
        # Save to CSV
        df_group.to_csv(output_path, index=False)
        grouped_dataframes[age_group] = df_group
        
        # Verify no missing values for this group
        missing_runs = df_group['Run'].isna().sum()
        missing_labels = df_group['label'].isna().sum()
        if missing_runs > 0 or missing_labels > 0:
            print(f"Warning: Found missing values in {age_group}!")
            print(f"  Missing Run IDs: {missing_runs}")
            print(f"  Missing labels: {missing_labels}")
    
    # Handle samples with missing age_group (if any)
    df_missing_age = df[df['age_group'].isna()]
    if len(df_missing_age) > 0:
        print(f"Warning: {len(df_missing_age)} samples have missing age_group and were excluded.")
    
    return grouped_dataframes

if __name__ == "__main__":
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'gadir_metadata.csv')
    
    # Run preprocessing
    preprocess_gadir(input_file)

