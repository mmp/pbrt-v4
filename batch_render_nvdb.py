import os
import sys
import tempfile
import shutil
import subprocess
import glob
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_frame_number(filename, basename):
    match = re.match(rf"{re.escape(basename)}(\d{{4}})\.nvdb$", os.path.basename(filename))
    return match.group(1) if match else None

def process_frame(nvdb_file, frame_id, basename, pbrt_dir, output_dir, base_pbrt_content):
    # nvdb_file = os.path.join(nvdb_dir, f"{basename}{frame_id}.nvdb")
    pbrt_file = os.path.join(pbrt_dir, f"{basename}{frame_id}.pbrt")
    output_file = os.path.join(output_dir, f"{basename}{frame_id}.exr")

    # Step 2: Replace placeholders in PBRT
    pbrt_content = base_pbrt_content.replace("__NVDB_PATH__", nvdb_file)
    pbrt_content = pbrt_content.replace("__OUTPUT_PATH__", output_file)

    with open(pbrt_file, "w") as f:
        f.write(pbrt_content)

    # Step 3: Render PBRT
    subprocess.run(["./build_release/pbrt", pbrt_file], check=True)

def render(nvdb_dir, basename, base_pbrt_file, output_dir, min_frame_id, max_frame_id):
    pbrt_dir = tempfile.mkdtemp(prefix="pbrt_")

    os.makedirs(output_dir, exist_ok=True)

    try:
        nvdb_files = sorted(glob.glob(os.path.join(nvdb_dir, f"{basename}????.nvdb")))
        if not nvdb_files:
            print(f"No .nvdb files found with basename '{basename}' in {vdb_dir}")
            return

        # Extract and validate frame IDs
        found_ids = []
        frame_map = {}

        for nvdb_file in nvdb_files:
            frame_id = extract_frame_number(nvdb_file, basename)
            if frame_id:
                if int(frame_id) >= min_frame_id and (int(frame_id) <= max_frame_id or max_frame_id < 0):
                    found_ids.append(int(frame_id))
                    frame_map[frame_id] = nvdb_file

        found_ids.sort()
        max_expected_id = found_ids[-1] + 1 if max_frame_id == -1 else max_frame_id
        expected_ids = list(range(min_frame_id, max_expected_id))
        missing_ids = [i for i in expected_ids if i not in found_ids]

        if missing_ids:
            print("⚠️  Missing frame IDs:")
            for mid in missing_ids:
                print(f"  {basename}{mid:04d}.vdb")

        # Load PBRT template
        with open(base_pbrt_file, "r") as f:
            base_pbrt_content = f.read()

        # Parallel rendering
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_frame, frame_map[frame_id], frame_id, basename,
                                pbrt_dir, output_dir, base_pbrt_content)
                for frame_id in frame_map
            ]
            for future in as_completed(futures):
                future.result()
        # for frame_id in found_ids:
        #     print(f"▶️  Rendering frame {frame_id:04d}")
        #     process_frame(frame_map[frame_id], frame_id, basename,
        #                   pbrt_dir, output_dir, base_pbrt_content)

    finally:
        print(f"\n🧹 Cleaning up temporary directories:\n  PBRT: {pbrt_dir}")
        shutil.rmtree(pbrt_dir)

if __name__ == "__main__":
    if len(sys.argv) not in [5, 7]:
        print("Usage: python batch_render_nvdb.py <nvdb_folder> <nvdb_basename> <base_pbrt_file> <output_exr_dir> <min_frame_id> <max_frame_id>")
        sys.exit(1)

    nvdb_folder = sys.argv[1]
    nvdb_basename = sys.argv[2]
    base_pbrt_file = sys.argv[3]
    output_exr_dir = sys.argv[4]
    min_frame_id = int(sys.argv[5]) if len(sys.argv) == 6 else 1
    max_frame_id = int(sys.argv[6]) if len(sys.argv) == 7 else -1

    render(nvdb_folder, nvdb_basename, base_pbrt_file, output_exr_dir, min_frame_id, max_frame_id)
