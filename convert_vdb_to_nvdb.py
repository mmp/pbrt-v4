import os
import sys
import tempfile
import shutil
import subprocess
import glob
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_frame_number(filename, basename):
    match = re.match(rf"{re.escape(basename)}(\d{{4}})\.vdb$", os.path.basename(filename))
    return match.group(1) if match else None

def process_frame(vdb_file, frame_id, basename, nvdb_dir): #pbrt_dir, output_dir, base_pbrt_content):
    nvdb_file = os.path.join(nvdb_dir, f"{basename}{frame_id}.nvdb")

    # Step 1: Convert to NVDB
    subprocess.run(["./build/vdb_to_nvdb", vdb_file, nvdb_file], check=True)

def convert(vdb_dir, basename, nvdb_output_folder):
    nvdb_dir = nvdb_output_folder

    os.makedirs(nvdb_dir, exist_ok=True)

    try:
        vdb_files = sorted(glob.glob(os.path.join(vdb_dir, f"{basename}????.vdb")))
        if not vdb_files:
            print(f"No .vdb files found with basename '{basename}' in {vdb_dir}")
            return

        # Extract and validate frame IDs
        found_ids = []
        frame_map = {}

        for vdb_file in vdb_files:
            frame_id = extract_frame_number(vdb_file, basename)
            if frame_id:
                found_ids.append(int(frame_id))
                frame_map[frame_id] = vdb_file

        found_ids.sort()
        expected_ids = list(range(1, found_ids[-1] + 1))
        missing_ids = [i for i in expected_ids if i not in found_ids]

        if missing_ids:
            print("⚠️  Missing frame IDs:")
            for mid in missing_ids:
                print(f"  {basename}{mid:04d}.vdb")

        # Parallel rendering
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_frame, frame_map[frame_id], frame_id, basename,
                                nvdb_dir)
                for frame_id in frame_map
            ]
            for future in as_completed(futures):
                future.result()

    finally:
        print(f"\n🧹 Cleaning up temporary directories:\n  NVDB: {nvdb_dir}\n)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python convert_and_render.py <vdb_folder> <vdb_basename> <nvdb_output_folder>")
        sys.exit(1)

    vdb_folder = sys.argv[1]
    vdb_basename = sys.argv[2]
    nvdb_output_folder = sys.argv[3]

    convert(vdb_folder, vdb_basename, nvdb_output_folder)
