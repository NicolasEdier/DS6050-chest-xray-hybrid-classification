"""
Data preprocessing for NIH ChestX-ray14 dataset
Handles stratified sampling, train/val/test splits, and image preprocessing
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm

class ChestXrayPreprocessor:
    """Preprocess NIH ChestX-ray14 dataset"""
    
    def __init__(self, data_dir, output_dir, random_seed=42):
        """
        Args:
            data_dir: Path to raw NIH ChestX-ray14 data
            output_dir: Path to save processed data
            random_seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 14 disease classes
        self.disease_classes = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
            'Pleural_Thickening', 'Hernia'
        ]
        
    def load_metadata(self):
        """Load and parse metadata CSV"""
        metadata_path = self.data_dir / 'Data_Entry_2017.csv'
        df = pd.read_csv(metadata_path)
        
        # Parse finding labels
        df['Finding Labels'] = df['Finding Labels'].apply(
            lambda x: x.split('|') if '|' in x else [x]
        )
        
        # Create binary labels for each disease
        for disease in self.disease_classes:
            df[disease] = df['Finding Labels'].apply(
                lambda x: 1 if disease in x else 0
            )
        
        # Add 'No Finding' label
        df['No Finding'] = df['Finding Labels'].apply(
            lambda x: 1 if x == ['No Finding'] else 0
        )
        
        return df
    
    def stratified_sampling(self, df, samples_per_disease=None):
        """
        Create stratified subset ensuring representation of all diseases
        
        Args:
            df: Full dataset dataframe
            samples_per_disease: Dict mapping disease to number of samples
                                If None, uses default values
        
        Returns:
            Sampled dataframe
        """
        if samples_per_disease is None:
            # Default sampling strategy from proposal
            samples_per_disease = {
                'No Finding': 2100,
                'Infiltration': 2200,
                'Effusion': 1800,
                'Atelectasis': 1600,
                'Nodule': 1200,
                'Mass': 1000,
                'Pneumothorax': 950,
                'Consolidation': 850,
                'Pleural_Thickening': 800,
                'Cardiomegaly': 750,
                'Emphysema': 700,
                'Edema': 650,
                'Fibrosis': 600,
                'Pneumonia': 550,
                'Hernia': 200
            }
        
        sampled_dfs = []
        
        for disease, n_samples in samples_per_disease.items():
            # Get images with this disease
            if disease == 'No Finding':
                disease_df = df[df['No Finding'] == 1]
            else:
                disease_df = df[df[disease] == 1]
            
            # Sample (with replacement if needed for rare diseases)
            if len(disease_df) < n_samples:
                print(f"Warning: {disease} has only {len(disease_df)} samples, "
                      f"requested {n_samples}. Using all available.")
                sampled = disease_df
            else:
                sampled = disease_df.sample(
                    n=n_samples, 
                    random_state=self.random_seed
                )
            
            sampled_dfs.append(sampled)
        
        # Combine and remove duplicates (some images have multiple diseases)
        combined_df = pd.concat(sampled_dfs, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['Image Index'])
        
        print(f"Total unique images after sampling: {len(combined_df)}")
        return combined_df
    
    def patient_level_split(self, df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
        """
        Split data at patient level to prevent leakage
        
        Args:
            df: Dataset dataframe
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
        
        Returns:
            train_df, val_df, test_df
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        # Get unique patients
        unique_patients = df['Patient ID'].unique()
        
        # First split: train vs (val + test)
        train_patients, temp_patients = train_test_split(
            unique_patients,
            test_size=(val_ratio + test_ratio),
            random_state=self.random_seed
        )
        
        # Second split: val vs test
        val_patients, test_patients = train_test_split(
            temp_patients,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=self.random_seed
        )
        
        # Create dataframes
        train_df = df[df['Patient ID'].isin(train_patients)]
        val_df = df[df['Patient ID'].isin(val_patients)]
        test_df = df[df['Patient ID'].isin(test_patients)]
        
        print(f"Train: {len(train_df)} images from {len(train_patients)} patients")
        print(f"Val: {len(val_df)} images from {len(val_patients)} patients")
        print(f"Test: {len(test_df)} images from {len(test_patients)} patients")
        
        return train_df, val_df, test_df
    
    def save_splits(self, train_df, val_df, test_df):
        """Save train/val/test splits as CSV files"""
        train_df.to_csv(self.output_dir / 'train.csv', index=False)
        val_df.to_csv(self.output_dir / 'val.csv', index=False)
        test_df.to_csv(self.output_dir / 'test.csv', index=False)
        print(f"Splits saved to {self.output_dir}")
        
    def get_class_weights(self, train_df):
        """Calculate inverse frequency class weights for training"""
        weights = {}
        total_samples = len(train_df)
        
        for disease in self.disease_classes:
            positive_samples = train_df[disease].sum()
            if positive_samples > 0:
                # Inverse frequency weighting
                weights[disease] = total_samples / (2 * positive_samples)
            else:
                weights[disease] = 1.0
        
        # Normalize weights
        mean_weight = np.mean(list(weights.values()))
        weights = {k: v / mean_weight for k, v in weights.items()}
        
        return weights
    
    def print_dataset_statistics(self, df, split_name="Dataset"):
        """Print statistics about the dataset"""
        print(f"\n{split_name} Statistics:")
        print(f"Total images: {len(df)}")
        print(f"Total patients: {df['Patient ID'].nunique()}")
        print(f"\nDisease distribution:")
        
        for disease in ['No Finding'] + self.disease_classes:
            if disease in df.columns:
                count = df[disease].sum()
                percentage = (count / len(df)) * 100
                print(f"  {disease:20s}: {count:5d} ({percentage:5.2f}%)")
        
        # Multi-label statistics
        df['num_diseases'] = df[self.disease_classes].sum(axis=1)
        print(f"\nMulti-label distribution:")
        print(f"  Images with 0 diseases: {(df['num_diseases'] == 0).sum()}")
        print(f"  Images with 1 disease:  {(df['num_diseases'] == 1).sum()}")
        print(f"  Images with 2+ diseases: {(df['num_diseases'] >= 2).sum()}")
        print(f"  Max diseases per image: {df['num_diseases'].max()}")


def main():
    """Main preprocessing pipeline"""
    # Set paths
    data_dir = './NIH_ChestXray'  # Update to your data directory
    output_dir = './processed_data'
    
    # Initialize preprocessor
    preprocessor = ChestXrayPreprocessor(data_dir, output_dir)
    
    # Load metadata
    print("Loading metadata...")
    df = preprocessor.load_metadata()
    print(f"Loaded {len(df)} images")
    
    # Create stratified subset
    print("\nCreating stratified subset...")
    subset_df = preprocessor.stratified_sampling(df)
    
    # Print statistics
    preprocessor.print_dataset_statistics(subset_df, "Subset")
    
    # Patient-level split
    print("\nSplitting into train/val/test...")
    train_df, val_df, test_df = preprocessor.patient_level_split(subset_df)
    
    # Print statistics for each split
    preprocessor.print_dataset_statistics(train_df, "Training Set")
    preprocessor.print_dataset_statistics(val_df, "Validation Set")
    preprocessor.print_dataset_statistics(test_df, "Test Set")
    
    # Save splits
    preprocessor.save_splits(train_df, val_df, test_df)
    
    # Calculate and save class weights
    class_weights = preprocessor.get_class_weights(train_df)
    print("\nClass weights for training:")
    for disease, weight in class_weights.items():
        print(f"  {disease:20s}: {weight:.3f}")
    
    # Save class weights
    import json
    with open(output_dir / 'class_weights.json', 'w') as f:
        json.dump(class_weights, f, indent=2)
    
    print(f"\nPreprocessing complete! Data saved to {output_dir}")


if __name__ == '__main__':
    main()
