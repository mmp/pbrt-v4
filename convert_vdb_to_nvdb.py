import openvdb
import argparse

def convert_vdb_to_nvdb(input_file, output_file):
    # Open the input VDB file
    grid = openvdb.read(input_file)

    # Write the grid to an output NanoVDB file
    openvdb.write(output_file, grid)
    print(f"Successfully converted {input_file} to {output_file}")

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Convert a VDB file to NanoVDB format.")
    parser.add_argument("input_file", help="Path to the input VDB file")
    parser.add_argument("output_file", help="Path to the output NanoVDB file")

    # Parse the arguments
    args = parser.parse_args()

    # Convert the VDB file to NanoVDB
    convert_vdb_to_nvdb(args.input_file, args.output_file)
