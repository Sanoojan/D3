import pandas as pd
import os
import random

def extract_identity(path):
    """Extract identity (e.g. id00475) from path"""
    parts = path.split('/')
    for p in parts:
        if p.startswith("id") and len(p) == 7:
            return p
    return None

def load_and_tag(csv_paths, label_type):
    """Load multiple CSVs and tag with label_type ('real' or 'fake')"""
    dfs = []
    for csv_path in csv_paths:
        # Detect header automatically
        with open(csv_path, 'r') as f:
            first_line = f.readline()
        has_header = not first_line.startswith("dataset/")  # crude but effective check

        df = pd.read_csv(csv_path, header=0 if has_header else None)
        df['identity'] = df.iloc[:, 0].apply(extract_identity)
        df['type'] = label_type
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True), has_header

def main(real_csvs, fake_csvs, output_dir="splits", split_ratio=0.8, seed=42):
    os.makedirs(output_dir, exist_ok=True)

    # Load datasets
    df_real, has_header_real = load_and_tag(real_csvs, "real")
    df_fake, has_header_fake = load_and_tag(fake_csvs, "fake")

    # Consistency check
    assert has_header_real == has_header_fake, "Real and fake CSVs must have consistent header structure."
    has_header = has_header_real

    # Collect unique identities across all
    all_ids = list(set(df_real['identity']).union(set(df_fake['identity'])))
    random.seed(seed)
    random.shuffle(all_ids)

    # Split IDs
    n_train = int(len(all_ids) * split_ratio)
    train_ids = set(all_ids[:n_train])
    test_ids = set(all_ids[n_train:])

    # Split datasets
    train_real = df_real[df_real['identity'].isin(train_ids)]
    test_real = df_real[df_real['identity'].isin(test_ids)]
    train_fake = df_fake[df_fake['identity'].isin(train_ids)]
    test_fake = df_fake[df_fake['identity'].isin(test_ids)]

    # Drop helper columns before saving
    for df in [train_real, test_real, train_fake, test_fake]:
        df.drop(columns=['identity', 'type'], inplace=True)

    # Save with or without headers
    header_option = True if has_header else False
    train_real.to_csv(os.path.join(output_dir, "train_real.csv"), index=False, header=header_option)
    test_real.to_csv(os.path.join(output_dir, "test_real.csv"), index=False, header=header_option)
    train_fake.to_csv(os.path.join(output_dir, "train_fake.csv"), index=False, header=header_option)
    test_fake.to_csv(os.path.join(output_dir, "test_fake.csv"), index=False, header=header_option)

    print(f"✅ Split complete! Saved in '{output_dir}'")
    print(f"  Train IDs: {len(train_ids)}, Test IDs: {len(test_ids)}")
    print(f"  Train real: {len(train_real)}, Test real: {len(test_real)}")
    print(f"  Train fake: {len(train_fake)}, Test fake: {len(test_fake)}")


if __name__ == "__main__":
    # Example usage — replace with your CSV paths
    real_csvs = [
        "dataset/DeepfakeDatasets/FakeAVCeleb/csv/RealVideo-FakeAudio.csv",
        "dataset/DeepfakeDatasets/FakeAVCeleb/csv/RealVideo-RealAudio.csv",
    ]
    fake_csvs = [
        "dataset/DeepfakeDatasets/FakeAVCeleb/csv/FakeVideo-RealAudio.csv",
        "dataset/DeepfakeDatasets/FakeAVCeleb/csv/FakeVideo-FakeAudio.csv",
    ]
    main(real_csvs, fake_csvs, output_dir="dataset/DeepfakeDatasets/FakeAVCeleb/csv", split_ratio=0.8)