#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/IO.h>
#include <openvdb/openvdb.h>
// #include <openvdb/tools/Interpolation.h>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " input.nvdb output.vdb\n";
        return 1;
    }

    const std::string inputFile = argv[1];
    const std::string outputFile = argv[2];

            // Load NanoVDB grid
    nanovdb::GridHandle<> handle = nanovdb::io::readGrid(inputFile);
    const auto* nanoGrid = handle.grid<float>();
    if (!nanoGrid) {
        std::cerr << "Error: No float grid found in NanoVDB file.\n";
        return 1;
    }

            // Initialize OpenVDB
    openvdb::initialize();

            // Create a new OpenVDB FloatGrid
    openvdb::FloatGrid::Ptr openGrid = openvdb::FloatGrid::create(nanoGrid->tree().background());
    openGrid->setName(nanoGrid->gridName());
    openGrid->setTransform(openvdb::math::Transform::createLinearTransform(nanoGrid->voxelSize()[0]));

    // Copy values
    auto acc = openGrid->getAccessor();

            // auto dim = nanoGrid->tree().background();
    const nanovdb::CoordBBox bbox = nanoGrid->indexBBox();

    for (int z = bbox.min().z(); z <= bbox.max().z(); ++z) {
        for (int y = bbox.min().y(); y <= bbox.max().y(); ++y) {
            for (int x = bbox.min().x(); x <= bbox.max().x(); ++x) {
                nanovdb::Coord ijk(x, y, z);
                float val = nanoGrid->tree().getValue(ijk);
                if (val != nanoGrid->tree().background()) {
                    // acc.setValue(openvdb::Coord(x, y, z), val);
                }
            }
        }
    }

            // Write to .vdb file
    openvdb::io::File file(outputFile);
    openvdb::GridPtrVec grids;
    grids.push_back(openGrid);
    file.write(grids);
    file.close();

    std::cout << "Conversion complete: " << outputFile << "\n";
    return 0;
}
