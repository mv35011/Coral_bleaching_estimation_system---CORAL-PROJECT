import pandas as pd
import argparse


def merge_data(original_file, missing_data_file, final_output_file):
    """
    Merges the original dataset with the newly fetched data.

    Args:
        original_file (str): Path to the master file with the empty rows.
        missing_data_file (str): Path to the newly downloaded data file.
        final_output_file (str): Path to save the final, complete dataset.
    """
    try:
        print(f"Loading original master file: {original_file}")
        df_master = pd.read_csv(original_file)

        print(f"Loading new data file: {missing_data_file}")
        df_missing = pd.read_csv(missing_data_file)
        print(f"\nOriginal file has {len(df_master):,} rows.")
        print(f"New data file has {len(df_missing):,} rows for Gulf of Mannar.")
        print("\nRemoving empty 'Gulf_of_Mannar' rows from the original file...")
        original_rows = len(df_master)
        df_master_filtered = df_master[df_master['location_name'] != 'Gulf_of_Mannar'].copy()
        rows_removed = original_rows - len(df_master_filtered)
        print(f"Removed {rows_removed:,} rows.")
        print("Concatenating the datasets...")
        df_final = pd.concat([df_master_filtered, df_missing], ignore_index=True)
        df_final = df_final.sort_values(['location_name', 'time']).reset_index(drop=True)
        df_final.to_csv(final_output_file, index=False)

        print(f"\n✅ Success! Final, complete dataset saved to '{final_output_file}'")
        print(f"The final dataset has {len(df_final):,} rows.")

    except FileNotFoundError as e:
        print(f"❌ ERROR: File not found. Please check the filenames. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge the main dataset with the fetched missing data.")
    parser.add_argument(
        'original_file',
        type=str,
        help="The original master CSV file (with missing data)."
    )
    parser.add_argument(
        'missing_data_file',
        type=str,
        help="The newly downloaded CSV for the missing location."
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='coral_data_COMPLETE.csv',
        help="The name for the final, merged output file."
    )
    args = parser.parse_args()

    merge_data(args.original_file, args.missing_data_file, args.output)
