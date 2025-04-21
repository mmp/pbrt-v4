#include <openvdb/openvdb.h>
#include <openvdb/io/Stream.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/util/IO.h>
#include <iostream>
#include <vector>
#include <memory>

// Function to print min/max density values
void printMinMaxDensity(const nanovdb::NanoGrid<float>& nanoGrid) {
    auto acc = nanoGrid.getAccessor();
    float minDensity = std::numeric_limits<float>::infinity();
    float maxDensity = -std::numeric_limits<float>::infinity();

            // Iterate through the active region of the grid to get min/max density values
            // Iterate over all possible coordinates within the bounding box
    nanovdb::CoordBBox bbox = acc.root().bbox();
    for (int x = bbox.min()[0]; x <= bbox.max()[0]; ++x) {
        for (int y = bbox.min()[1]; y <= bbox.max()[1]; ++y) {
            for (int z = bbox.min()[2]; z <= bbox.max()[2]; ++z) {
                // Create a coordinate object for this voxel
                nanovdb::Coord coord(x, y, z);

                        // Check if the voxel is active (non-zero density)
                if (acc.isActive(coord)) {
                    // Access the value at this voxel
                    float value = acc.getValue(coord);

                            // Update min/max density values
                    minDensity = std::min(minDensity, value);
                    maxDensity = std::max(maxDensity, value);
                }
            }
        }
    }

    std::cout << "Min Density: " << minDensity << std::endl;
    std::cout << "Max Density: " << maxDensity << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " input.nvdb output.vdb\n";
        return 1;
    }

    const std::string inputFile = argv[1];
    const std::string outputFile = argv[2];
    try {
        openvdb::initialize();

                // Open the VDB file
        openvdb::io::File file(inputFile);
        file.open();

        std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> nanoHandles;

                // Iterate through all grid names in the file
        for (auto nameIter = file.beginName(); nameIter != file.endName(); ++nameIter) {
            const std::string& gridName = nameIter.gridName();
            // std::cout << "Reading grid: " << gridName << std::endl;

                    // Read the grid from file
            openvdb::GridBase::Ptr baseGrid = file.readGrid(gridName);

                    // Try casting to FloatGrid
            auto floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
            if (!floatGrid) {
                std::cerr << "Warning: Grid \"" << gridName << "\" is not a FloatGrid. Skipping." << std::endl;
                continue;
            }

                    // Convert to NanoVDB
            auto handle = nanovdb::tools::createNanoGrid(*floatGrid);

            nanoHandles.push_back(std::move(handle));

                    // Accessors (example)
            auto* dstGrid = nanoHandles.back().grid<float>();
            auto dstAcc = dstGrid->getAccessor();
            auto srcAcc = floatGrid->getAccessor();

                    // printMinMaxDensity(*handle.grid<float>());

                    // float minDensity = std::numeric_limits<float>::infinity();
                    // float maxDensity = -std::numeric_limits<float>::infinity();

                    // // Iterate through the active region of the grid to get min/max density values
                    // // Iterate over all possible coordinates within the bounding box
                    // nanovdb::CoordBBox bbox = dstAcc.root().bbox();
                    // for (int x = bbox.min()[0]; x <= bbox.max()[0]; ++x) {
                    //     for (int y = bbox.min()[1]; y <= bbox.max()[1]; ++y) {
                    //         for (int z = bbox.min()[2]; z <= bbox.max()[2]; ++z) {
                    //             // Create a coordinate object for this voxel
                    //             nanovdb::Coord coord(x, y, z);

                    //             // Check if the voxel is active (non-zero density)
                    //             if (dstAcc.isActive(coord)) {
                    //                 // Access the value at this voxel
                    //                 float value = dstAcc.getValue(coord);
                    //                 // dstAcc.setValue(coord, value + 45.f);
                    //                 // dstGrid->getAccessor().set<float>(coord, value);

                    //                 // Update min/max density values
                    //                 minDensity = std::min(minDensity, value);
                    //                 maxDensity = std::max(maxDensity, value);
                    //             }
                    //         }
                    //     }
                    // }

                    // std::cout << "Min " << gridName << ": " << minDensity << std::endl;
                    // std::cout << "Max " << gridName << ": " << maxDensity << std::endl;
        }

        nanovdb::io::writeGrids(outputFile, nanoHandles);

        file.close();
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }

    return 0;
}
