#include <pbrt/wavefront/integrator.h>

#include <pbrt/media.h>

#include <type_traits>
#ifdef PBRT_BUILD_GPU_RENDERER
#include <pbrt/gpu/util.h>
#endif
namespace pbrt
{
    void WavefrontPathIntegrator::OutputRayDataToFiles()
    {
        if (!outputToFile || !outputRayDataFile || !inputRayDataFile) return;
        // INFO: Wait for GPU synchronization
#ifdef PBRT_BUILD_GPU_RENDERER
        GPUWait();
#endif
        // Copy data from GPU to CPU if using GPU rendering
        // The SOA data will be automatically synchronized when accessed on CPU

        //For now, we just set a limit on the size of the files, set to 1GB for now
                // Ensure file pointers are positioned at the end so writes append.
        const uint64_t kMaxBytes = 1024ull * 1024ull * 1024ull;

        auto outPos = outputRayDataFile->tellp();
        auto inPos  = inputRayDataFile->tellp();

        // If tellp() is not valid, try seeking to end and re-query.
        if (outPos == std::streampos(-1) || inPos == std::streampos(-1)) {
            outputRayDataFile->seekp(0, std::ios::end);
            inputRayDataFile->seekp(0, std::ios::end);
            outPos = outputRayDataFile->tellp();
            inPos  = inputRayDataFile->tellp();
        }

        // If either file is already >= 1 GiB, don't write anything.
        if ((outPos != std::streampos(-1) && static_cast<uint64_t>(outPos) >= kMaxBytes) ||
            (inPos  != std::streampos(-1) && static_cast<uint64_t>(inPos)  >= kMaxBytes)) {
            LOG_VERBOSE("Files are larger than 1GB, skipping outputting...\n");
            return;
        }

        // Make sure we append to the end.
        outputRayDataFile->seekp(0, std::ios::end);
        inputRayDataFile->seekp(0, std::ios::end);

        Bounds2i pixelBounds = film.PixelBounds();
        Vector2i resolution = pixelBounds.Diagonal();

        // Loop over all pixel samples and write ray data to file
        for (int pixelIndex = 0; pixelIndex < maxQueueSize; ++pixelIndex)
        {
            //TODO: Consider using a cudaMemcpy(Async) to perform this so its slightly quicker.

            Point3f iRayo = inputRayData.rayo[pixelIndex];
            Vector3f iRayd = inputRayData.rayd[pixelIndex];
            int iDepth = inputRayData.depth[pixelIndex];
            int iSampleIdx = inputRayData.sampleIdx[pixelIndex];

            // Read data from SOA structure
            Point3f oRayo = outputRayData.rayo[pixelIndex];
            Vector3f oRayd = outputRayData.rayd[pixelIndex];
            int oDepth = outputRayData.depth[pixelIndex];
            int oSampleIdx = outputRayData.sampleIdx[pixelIndex];
            SampledSpectrum L = outputRayData.L[pixelIndex];
            SampledWavelengths lambda = outputRayData.lambda[pixelIndex];
            //TODO: Set lambda in cpu side

            
            //TODO: Check if this is the correct way to get the pixelrgb
            // RGB rgb = film.GetPixelRGB(p + pixelBounds.pMin);
            RGB rgb = film.ToOutputRGB(L, lambda);

            std::string inputLine = StringPrintf("Pixel (o: %s, d: %s) PixelIdx (%d) Sample (%d) InitialDepth (%d)\n",
                iRayo.ToString(), iRayd.ToString(), pixelIndex, iSampleIdx, iDepth
            );
            inputRayDataFile->write(inputLine.c_str(), inputLine.length());

            std::string outputLine = StringPrintf("Pixel (o: %s, d: %s) PixelIdx (%d) Sample (%d) FinalDepth (%d) Luminance (%s) RGB (%s)\n",
                oRayo.ToString(), oRayd.ToString(), pixelIndex, oSampleIdx, oDepth, L.ToString(), rgb.ToString());
            outputRayDataFile->write(outputLine.c_str(), outputLine.length());
        }

        //NOTE: Proposed Async solution
        //             Bounds2i pixelBounds = film.PixelBounds();

        //     // Allocate pinned CPU memory for faster async transfers
        //     Point2i* pPixel_cpu;
        //     Point3f* rayo_cpu;
        //     Vector3f* rayd_cpu;
        //     int* depth_cpu;
        //     int* sampleIdx_cpu;

        // #ifdef PBRT_BUILD_GPU_RENDERER
        //     if (Options->useGPU) {
        //         // Allocate pinned (page-locked) host memory for async transfer
        //         cudaMallocHost(&pPixel_cpu, maxQueueSize * sizeof(Point2i));
        //         cudaMallocHost(&rayo_cpu, maxQueueSize * sizeof(Point3f));
        //         cudaMallocHost(&rayd_cpu, maxQueueSize * sizeof(Vector3f));
        //         cudaMallocHost(&depth_cpu, maxQueueSize * sizeof(int));
        //         cudaMallocHost(&sampleIdx_cpu, maxQueueSize * sizeof(int));

        //         // Async copy from GPU to CPU (non-blocking)
        //         cudaMemcpyAsync(pPixel_cpu, outputRayData.pPixel, 
        //                     maxQueueSize * sizeof(Point2i), cudaMemcpyDeviceToHost);
        //         cudaMemcpyAsync(rayo_cpu, outputRayData.rayo, 
        //                     maxQueueSize * sizeof(Point3f), cudaMemcpyDeviceToHost);
        //         cudaMemcpyAsync(rayd_cpu, outputRayData.rayd, 
        //                     maxQueueSize * sizeof(Vector3f), cudaMemcpyDeviceToHost);
        //         cudaMemcpyAsync(depth_cpu, outputRayData.depth, 
        //                     maxQueueSize * sizeof(int), cudaMemcpyDeviceToHost);
        //         cudaMemcpyAsync(sampleIdx_cpu, outputRayData.sampleIdx, 
        //                     maxQueueSize * sizeof(int), cudaMemcpyDeviceToHost);

        //         // Synchronize only the copy stream (not the render stream)
        //         cudaStreamSynchronize(0);
        //     } else
        // #endif
        //     {
        //         // CPU rendering - just use the pointers directly
        //         pPixel_cpu = outputRayData.pPixel;
        //         rayo_cpu = outputRayData.rayo;
        //         rayd_cpu = outputRayData.rayd;
        //         depth_cpu = outputRayData.depth;
        //         sampleIdx_cpu = outputRayData.sampleIdx;
        //     }


        //         *outputRayDataFile << "Pixel: (" << pPixel.x << ", " << pPixel.y << ") "
        //                         << "Sample: " << sampleIdx_cpu[pixelIndex] << " "
        //                         << "Depth: " << depth_cpu[pixelIndex] << " "
        //                         << "Origin: (" << rayo_cpu[pixelIndex].x << ", " 
        //                         << rayo_cpu[pixelIndex].y << ", " << rayo_cpu[pixelIndex].z << ") "
        //                         << "Direction: (" << rayd_cpu[pixelIndex].x << ", " 
        //                         << rayd_cpu[pixelIndex].y << ", " << rayd_cpu[pixelIndex].z << ")\n";
        //     }

        //     outputRayDataFile->flush();

        // #ifdef PBRT_BUILD_GPU_RENDERER
        //     if (Options->useGPU) {
        //         // Free pinned memory
        //         cudaFreeHost(pPixel_cpu);
        //         cudaFreeHost(rayo_cpu);
        //         cudaFreeHost(rayd_cpu);
        //         cudaFreeHost(depth_cpu);
        //         cudaFreeHost(sampleIdx_cpu);
        //     }
        // #endif

            // Flush the file to ensure data is written
        outputRayDataFile->flush();
    }


}