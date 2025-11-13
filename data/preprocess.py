"""
Data preprocessing for NIH ChestX-ray14 dataset
Downloads images and metadata, creates stratified subset
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import urllib.request
import tarfile
from tqdm import tqdm

class ChestXrayPreprocessor:
    """Download and preprocess NIH ChestX-ray14 dataset"""
    
    def __init__(self, data_dir='./NIH_ChestXray', output_dir='./processed_data', random_seed=42):
        """
        Args:
            data_dir: Path to store raw NIH ChestX-ray14 data
            output_dir: Path to save processed data
            random_seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 14 disease classes
        self.disease_classes = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
            'Pleural_Thickening', 'Hernia'
        ]
        
        # Image download links
        self.image_links = [
            'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
            'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
            'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
            'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
            'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
            'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
            'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
            'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
            'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
            'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
            'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
            'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
        ]
        
        # Metadata CSV link
        self.metadata_link = 'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz'
    
    def download_images(self, skip_if_exists=True):
        """
        Download all image tar.gz files
        
        Args:
            skip_if_exists: Skip download if files already exist
        """
        print("="*70)
        print("DOWNLOADING NIH CHESTX-RAY14 IMAGES")
        print("="*70)
        print(f"Download location: {self.data_dir}")
        print(f"Total files to download: {len(self.image_links)}")
        print("This will take a while (dataset is ~45GB)...\n")
        
        for idx, link in enumerate(self.image_links, 1):
            filename = f'images_{idx:02d}.tar.gz'
            filepath = self.data_dir / filename
            
            # Skip if already exists
            if skip_if_exists and filepath.exists():
                print(f"✓ {filename} already exists, skipping...")
                continue
            
            print(f"Downloading {filename} ({idx}/{len(self.image_links)})...")
            try:
                urllib.request.urlretrieve(link, filepath)
                print(f"  ✓ Downloaded {filename}")
            except Exception as e:
                print(f"  ✗ Error downloading {filename}: {e}")
                continue
        
        print("\n✓ Image download complete!")
    
    def extract_images(self, skip_if_exists=True):
        """
        Extract all tar.gz files
        
        Args:
            skip_if_exists: Skip extraction if directory already exists
        """
        print("\n" + "="*70)
        print("EXTRACTING IMAGE FILES")
        print("="*70)
        
        for idx in range(1, len(self.image_links) + 1):
            filename = f'images_{idx:02d}.tar.gz'
            filepath = self.data_dir / filename
            
            if not filepath.exists():
                print(f"⚠ {filename} not found, skipping extraction...")
                continue
            
            # Check if already extracted
            extract_dir = self.data_dir / f'images_{idx:02d}'
            if skip_if_exists and extract_dir.exists():
                print(f"✓ {filename} already extracted, skipping...")
                continue
            
            print(f"Extracting {filename}...")
            try:
                with tarfile.open(filepath, 'r:gz') as tar:
                    tar.extractall(path=self.data_dir)
                print(f"  ✓ Extracted {filename}")
            except Exception as e:
                print(f"  ✗ Error extracting {filename}: {e}")
                continue
        
        print("\n✓ Extraction complete!")
    
    def download_metadata(self):
        """Download metadata CSV file"""
        print("\n" + "="*70)
        print("DOWNLOADING METADATA")
        print("="*70)
        
        metadata_file = self.data_dir / 'Data_Entry_2017.csv'
        
        if metadata_file.exists():
            print("✓ Metadata file already exists")
            return metadata_file
        
        # The metadata CSV is available at a direct link
        metadata_url = 'https://nihcc.box.com/shared/static/e6w2b6h3fj5sj4xh8w8m9e2c6v2r8r5b.csv'
        
        print(f"Downloading metadata CSV...")
        try:
            # Try direct download
            urllib.request.urlretrieve(metadata_url, metadata_file)
            print("✓ Metadata downloaded")
        except:
            print("⚠ Could not download metadata automatically")
            print("Please download manually from:")
            print("https://nihcc.box.com/v/ChestXray-NIHCC/file/220660789610")
            print(f"Save as: {metadata_file}")
            raise FileNotFoundError("Metadata file not found. Please download manually.")
        
        return metadata_file
    
    def load_metadata(self):
        """Load and parse metadata CSV"""
        metadata_path = self.data_dir / 'Data_Entry_2017.csv'
        
        if not metadata_path.exists():
            print("⚠ Metadata CSV not found!")
            print("Please download Data_Entry_2017.csv from:")
            print("https://nihcc.box.com/v/ChestXray-NIHCC")
            print(f"And place it in: {self.data_dir}")
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        
        print(f"\nLoading metadata from {metadata_path}...")
        df = pd.read_csv(metadata_path)
        
        # Parse finding labels
        df['Finding Labels'] = df['Finding Labels'].apply(
            lambda x: x.split('|') if '|' in str(x) else [str(x)]
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
        
        print(f"✓ Loaded metadata for {len(df)} images")
        return df
    
    def verify_images_exist(self, df):
        """Verify that image files exist for the metadata"""
        print("\nVerifying image files...")
        
        # Get all image files
        all_images = []
        for images_dir in self.data_dir.glob('images_*'):
            if images_dir.is_dir():
                image_files = list(images_dir.glob('images/*.png'))
                all_images.extend([f.name for f in image_files])
        
        print(f"Found {len(all_images)} image files on disk")
        
        # Filter dataframe to only include existing images
        df_filtered = df[df['Image Index'].isin(all_images)].copy()
        
        print(f"Metadata entries with existing images: {len(df_filtered)}/{len(df)}")
        
        if len(df_filtered) == 0:
            print("\n⚠ WARNING: No images found!")
            print("Make sure images are extracted to:")
            print(f"  {self.data_dir}/images_01/images/")
            print(f"  {self.data_dir}/images_02/images/")
            print("  etc.")
            raise FileNotFoundError("No image files found!")
        
        return df_filtered
    
    def stratified_sampling(self, df, samples_per_disease=None):
        """
        Create stratified subset ensuring representation of all diseases
        
        Args:
            df: Full dataset dataframe
            samples_per_disease: Dict mapping disease to number of samples
        
        Returns:
            Sampled dataframe
        """
        if samples_per_disease is None:
            # Default sampling strategy
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
            
            # Sample
            if len(disease_df) < n_samples:
                print(f"⚠ {disease}: only {len(disease_df)} available, requested {n_samples}")
                sampled = disease_df
            else:
                sampled = disease_df.sample(n=n_samples, random_state=self.random_seed)
            
            sampled_dfs.append(sampled)
        
        # Combine and remove duplicates
        combined_df = pd.concat(sampled_dfs, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['Image Index'])
        
        print(f"\n✓ Stratified sampling complete: {len(combined_df)} unique images")
        return combined_df
    
    def patient_level_split(self, df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
        """Split data at patient level to prevent leakage"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
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
        
        print(f"\n✓ Data split complete:")
        print(f"  Train: {len(train_df)} images from {len(train_patients)} patients")
        print(f"  Val:   {len(val_df)} images from {len(val_patients)} patients")
        print(f"  Test:  {len(test_df)} images from {len(test_patients)} patients")
        
        return train_df, val_df, test_df
    
    def save_splits(self, train_df, val_df, test_df):
        """Save train/val/test splits as CSV files"""
        train_df.to_csv(self.output_dir / 'train.csv', index=False)
        val_df.to_csv(self.output_dir / 'val.csv', index=False)
        test_df.to_csv(self.output_dir / 'test.csv', index=False)
        print(f"\n✓ Splits saved to {self.output_dir}/")
    
    def get_class_weights(self, train_df):
        """Calculate inverse frequency class weights"""
        weights = {}
        total_samples = len(train_df)
        
        for disease in self.disease_classes:
            positive_samples = train_df[disease].sum()
            if positive_samples > 0:
                weights[disease] = total_samples / (2 * positive_samples)
            else:
                weights[disease] = 1.0
        
        # Normalize
        mean_weight = np.mean(list(weights.values()))
        weights = {k: v / mean_weight for k, v in weights.items()}
        
        return weights
    
    def print_dataset_statistics(self, df, split_name="Dataset"):
        """Print statistics about the dataset"""
        print(f"\n{split_name} Statistics:")
        print(f"  Total images: {len(df)}")
        print(f"  Total patients: {df['Patient ID'].nunique()}")
        print(f"\n  Disease distribution:")
        
        for disease in ['No Finding'] + self.disease_classes:
            if disease in df.columns:
                count = df[disease].sum()
                percentage = (count / len(df)) * 100
                print(f"    {disease:20s}: {count:5d} ({percentage:5.2f}%)")


def main():
    """Main preprocessing pipeline"""
    print("\n" + "="*70)
    print("NIH CHESTX-RAY14 PREPROCESSING PIPELINE")
    print("="*70)
    
    # Initialize preprocessor
    preprocessor = ChestXrayPreprocessor()
    
    # Step 1: Download images (this will take a while!)
    print("\nStep 1: Download images")
    print("⚠ This will download ~45GB of data. Continue? (y/n): ", end='')
    response = input().lower()
    
    if response == 'y':
        preprocessor.download_images()
        preprocessor.extract_images()
    else:
        print("Skipping download. Make sure images are already downloaded!")
    
    # Step 2: Load metadata
    print("\nStep 2: Load metadata")
    try:
        df = preprocessor.load_metadata()
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease download Data_Entry_2017.csv manually and re-run.")
        return
    
    # Step 3: Verify images exist
    print("\nStep 3: Verify image files")
    df = preprocessor.verify_images_exist(df)
    
    # Step 4: Create stratified subset
    print("\nStep 4: Create stratified subset")
    subset_df = preprocessor.stratified_sampling(df)
    preprocessor.print_dataset_statistics(subset_df, "Subset")
    
    # Step 5: Patient-level split
    print("\nStep 5: Split into train/val/test")
    train_df, val_df, test_df = preprocessor.patient_level_split(subset_df)
    
    preprocessor.print_dataset_statistics(train_df, "Training Set")
    preprocessor.print_dataset_statistics(val_df, "Validation Set")
    preprocessor.print_dataset_statistics(test_df, "Test Set")
    
    # Step 6: Save splits
    print("\nStep 6: Save splits")
    preprocessor.save_splits(train_df, val_df, test_df)
    
    # Step 7: Calculate class weights
    print("\nStep 7: Calculate class weights")
    class_weights = preprocessor.get_class_weights(train_df)
    print("\n  Class weights:")
    for disease, weight in class_weights.items():
        print(f"    {disease:20s}: {weight:.3f}")
    
    # Save class weights
    import json
    with open(preprocessor.output_dir / 'class_weights.json', 'w') as f:
        json.dump(class_weights, f, indent=2)
    
    print("\n" + "="*70)
    print("✓ PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"\nProcessed data saved to: {preprocessor.output_dir}")
    print(f"Raw images location: {preprocessor.data_dir}")
    print("\nYou can now run: python run_experiment.py --mode train")


if __name__ == '__main__':
    main()