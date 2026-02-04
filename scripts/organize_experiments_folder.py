import shutil
from pathlib import Path

# --- CONFIGURATION ---
SOURCE_DIR = Path("./local_experiments")
DEST_BASE = Path("./terumo_experiments/no-normalization")

# Define specific destination subfolders
DEST_ATT = DEST_BASE / "att-metric"
DEST_SEB = DEST_BASE / "seb"

def organize_experiments(dry_run=False):
    """
    Moves experiment contents to a structured destination based on folder names.
    dry_run: If True, only prints what would happen without moving files.
    """
    if not SOURCE_DIR.exists():
        print(f"Source directory {SOURCE_DIR} not found.")
        return

    # Iterate through model folders (virchow, phikon, etc.)
    for model_path in SOURCE_DIR.iterdir():
        if not model_path.is_dir():
            continue

        model_name = model_path.name

        # Iterate through dataset folders (bracs, lung-colon, etc.)
        for dataset_path in model_path.iterdir():
            if not dataset_path.is_dir():
                continue

            dataset_name = dataset_path.name

            # Iterate through actual experiment result folders
            for exp_folder in dataset_path.iterdir():
                if not exp_folder.is_dir():
                    continue

                folder_name = exp_folder.name
                target_root = None

                # 1. Determine destination based on naming pattern
                if folder_name.startswith("2_branch_mlp"):
                    target_root = DEST_ATT
                elif "_fsl_" in folder_name:
                    target_root = DEST_SEB

                # 2. If a match is found, move the contents
                if target_root:
                    # Construct: terumo_experiments/<category>/<model>/<dataset>/
                    final_dest = target_root / model_name / dataset_name
                    
                    if not dry_run:
                        final_dest.mkdir(parents=True, exist_ok=True)

                    print(f"Moving contents of: {exp_folder} -> {final_dest}")

                    # Move every item inside the timestamped folder to the final destination
                    for item in exp_folder.iterdir():
                        dest_item = final_dest / item.name
                        
                        if not dry_run:
                            # Use shutil.move to handle both files and directories
                            # Note: If file exists, this will raise an error. 
                            shutil.move(str(item), str(dest_item))
                    
                    # Optional: Remove the now-empty timestamped folder
                    if not dry_run:
                        try:
                            exp_folder.rmdir()
                        except OSError:
                            print(f"Note: Could not delete {exp_folder} (might not be empty)")

if __name__ == "__main__":
    # SET TO FALSE TO ACTUALLY MOVE FILES
    DRY_RUN_MODE = False 
    
    if DRY_RUN_MODE:
        print("--- RUNNING IN DRY RUN MODE (No files will be moved) ---")
    
    organize_experiments(dry_run=DRY_RUN_MODE)
    
    if DRY_RUN_MODE:
        print("--- DRY RUN COMPLETE. Set DRY_RUN_MODE = False to execute. ---")