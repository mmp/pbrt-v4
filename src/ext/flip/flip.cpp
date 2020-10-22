// Derived from FLIP reference implementation: https://research.nvidia.com/sites/default/files/pubs/2020-07_FLIP/flip-cpp.zip

/***************************************************************************
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
***************************************************************************/

// FLIP: A Difference Evaluator for Alternating Images
// High Performance Graphics, 2020.
// by Pontus Andersson, Jim Nilsson, Tomas Akenine-Moller, Magnus Oskarsson, Kalle Astrom, and Mark D. Fairchild
//
// Pointer to our paper: https://research.nvidia.com/publication/2020-07_FLIP
// code by Pontus Andersson, Jim Nilsson, and Tomas Akenine-Moller

// pooling.h

#ifdef WIN32
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <cassert>

namespace flip_detail {

class histogram
{
private:
    double mMinValue, mMaxValue;
    size_t mValueCount, mErrorValueCount;
    double mBucketSize;
    std::vector<size_t> mvBuckets;
    size_t mBucketIdRange[2];
    size_t mBucketIdMax;

public:
    histogram(size_t buckets, double minValue = 0.0, double maxValue = 1.0)
    {
        this->init(buckets, minValue, maxValue);
    }

    void init(size_t buckets, double minValue, double maxValue, size_t value = 0)
    {
        this->mMinValue = minValue;
        this->mMaxValue = maxValue;
        this->mValueCount = 0;
        this->mErrorValueCount = 0;
        this->mBucketIdRange[0] = std::string::npos;
        this->mBucketIdRange[1] = 0;
        this->resize(buckets);
        this->mvBuckets.resize(buckets, value);
    }

    double getBucketSize() const { return mBucketSize; }
    size_t getBucketIdMin() const { return mBucketIdRange[0]; }
    size_t getBucketIdMax() const { return mBucketIdRange[1]; }
    size_t getBucketValue(size_t bucketId) const { return mvBuckets[bucketId]; }
    size_t size() const { return mvBuckets.size(); }
    double getMinValue() const { return this->mMinValue; }
    double getMaxValue() const { return this->mMaxValue; }
    double getBucketStep() const { return (this->mMaxValue - this->mMinValue) / this->mvBuckets.size(); }

    void clear(size_t value = 0)
    {
        this->mvBuckets.resize(mvBuckets.size(), value);
    }

    void resize(size_t buckets)
    {
        this->mBucketSize = (this->mMaxValue - this->mMinValue) / buckets;
        this->mvBuckets.resize(buckets);
    }

    size_t valueBucketId(double value) const
    {
        if (value < this->mMinValue || value > this->mMaxValue)
            return std::string::npos;

        size_t bucketId = size_t(value / this->mBucketSize);

        if (bucketId == this->mvBuckets.size())
        {
            bucketId--;
        }
        return bucketId;
    }

    void inc(double value, size_t amount = 1)
    {
        size_t bucketId = valueBucketId(value);
        if (bucketId != std::string::npos)
        {
            this->mvBuckets[bucketId] += amount;
            this->mValueCount += amount;
            this->mBucketIdRange[0] = std::min(this->mBucketIdRange[0], bucketId);
            this->mBucketIdRange[1] = std::max(this->mBucketIdRange[1], bucketId);
        }
        else
        {
            mErrorValueCount += amount;
        }
    }

    std::string toPython(size_t numPixels, float ppd, double meanValue, double stddevValue, double maxValue, double minValue, double weightedMedian, double firstWeightedQuartile, double thirdWeightedQuartile, const bool optionLog) const
    {
        std::stringstream ss;

        double bucketStep = getBucketStep();

        //  imports
        ss << "import matplotlib.pyplot as plt\n";
        ss << "import sys\n";
        ss << "import numpy as np\n";
        ss << "from matplotlib.ticker import (MultipleLocator)\n\n";

        ss << "dimensions = (25, 15)  #  centimeters\n\n";

        ss << "lineColor = 'blue'\n";
        ss << "fillColor = 'lightblue'\n";
        ss << "meanLineColor = 'gray'\n";
        ss << "stddevLineColor = 'orange'\n";
        ss << "weightedMedianLineColor = 'red'\n";
        ss << "quartileLineColor = 'purple'\n";
        ss << "cogLineColor = 'green'\n";
        ss << "fontSize = 14\n";
        ss << "numPixels = " << numPixels << "\n\n";

        ss << "ppd = " << ppd << "\n";
        ss << "meanValue = " << meanValue << "\n";
        ss << "stddevValue = " << stddevValue << "\n";
        ss << "maxValue = " << maxValue << "\n";
        ss << "minValue = " << minValue << "\n\n";
        ss << "weightedMedianValue = " << weightedMedian << "\n\n";
        ss << "firstWeightedQuartileValue = " << firstWeightedQuartile << "\n\n";
        ss << "thirdWeightedQuartileValue = " << thirdWeightedQuartile << "\n\n";

        //  X-axis
        ss << "dataX = [";
        for (size_t bucketId = 0; bucketId < this->mvBuckets.size(); bucketId++)
        {
            ss << (bucketId > 0 ? ", " : "");
            ss << bucketStep * bucketId + 0.5 * bucketStep;
        }
        ss << "]\n\n";

        //  FLIP histogram
        ss << "dataFLIP = [";
        for (size_t bucketId = 0; bucketId < this->mvBuckets.size(); bucketId++)
        {
            ss << (bucketId > 0 ? ", " : "");
            ss << this->mvBuckets[bucketId];
        }
        ss << "]\n\n";

        //  Weighted FLIP histogram
        ss << "bucketStep = " << bucketStep << "\n";
        ss << "weightedDataFLIP = np.empty(" << this->mvBuckets.size() << ")\n";
        ss << "moments = np.empty(" << this->mvBuckets.size() << ")\n";
        ss << "for i in range(" << this->mvBuckets.size() << ") :\n";
        ss << "\tweight = (i + 0.5) * bucketStep\n";
        ss << "\tweightedDataFLIP[i] = dataFLIP[i] * weight\n";
        ss << "\tmoments[i] = dataFLIP[i] * weight * weight\n";
        ss << "cog = sum(moments) / sum(weightedDataFLIP)\n";
        ss << "weightedDataFLIP /= (numPixels /(1024 * 1024))  # normalized with the number of megapixels in the image\n\n";
        if (optionLog)
        {
            ss << "for i in range(" << this->mvBuckets.size() << ") :\n";
            ss << "\tif weightedDataFLIP[i] > 0 :\n";
            ss << "\t\tweightedDataFLIP[i] = np.log10(weightedDataFLIP[i])  # avoid log of zero\n\n";
        }

        ss << "maxY = max(weightedDataFLIP)\n\n";

        ss << "sumWeightedDataFLIP = sum(weightedDataFLIP)\n\n";

        ss << "font = { 'family' : 'serif', 'style' : 'normal', 'weight' : 'normal', 'size' : fontSize }\n";
        ss << "lineHeight = fontSize / (dimensions[1] * 15)\n";
        ss << "plt.rc('font', **font)\n";
        ss << "fig = plt.figure()\n";
        ss << "axes = plt.axes()\n";
        ss << "axes.xaxis.set_minor_locator(MultipleLocator(0.1))\n";
        ss << "axes.xaxis.set_major_locator(MultipleLocator(0.2))\n\n";

        ss << "fig.set_size_inches(dimensions[0] / 2.54, dimensions[1] / 2.54)\n"; 

        if(optionLog)
            ss << "axes.set(title = 'Weighted \\uA7FBLIP Histogram', xlabel = '\\uA7FBLIP error', ylabel = 'log(weighted \\uA7FBLIP sum per megapixel)')\n\n";
        else
            ss << "axes.set(title = 'Weighted \\uA7FBLIP Histogram', xlabel = '\\uA7FBLIP error', ylabel = 'Weighted \\uA7FBLIP sum per megapixel')\n\n";

        ss << "plt.bar(dataX, weightedDataFLIP, width = " << bucketStep << ", color = fillColor, edgeColor = lineColor, lineWidth = 0.3)\n\n";

        ss << "plt.text(0.99, 1.0 - 1 * lineHeight, 'PPD: ' + str(f'{ppd:.1f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes, color='black')\n\n";
        ss << "plt.text(0.99, 1.0 - 2 * lineHeight, 'Weighted median: ' + str(f'{weightedMedianValue:.4f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes, color=weightedMedianLineColor)\n\n";
        ss << "plt.text(0.99, 1.0 - 3 * lineHeight, 'Mean: ' + str(f'{meanValue:.4f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes, color=meanLineColor)\n\n";
        ss << "plt.text(0.99, 1.0 - 4 * lineHeight, '1st weighted quartile: ' + str(f'{firstWeightedQuartileValue:.4f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes, color=quartileLineColor)\n\n";
        ss << "plt.text(0.99, 1.0 - 5 * lineHeight, '3rd weighted quartile: ' + str(f'{thirdWeightedQuartileValue:.4f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes, color=quartileLineColor)\n\n";
        ss << "plt.text(0.99, 1.0 - 6 * lineHeight, 'Min: ' + str(f'{minValue:.4f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes)\n";
        ss << "plt.text(0.99, 1.0 - 7 * lineHeight, 'Max: ' + str(f'{maxValue:.4f}'), ha = 'right', fontsize = fontSize, transform = axes.transAxes)\n";

        ss << "axes.set_xlim(0.0, 1.0)\n";
        ss << "axes.set_ylim(0.0, maxY * 1.05)\n";
        ss << "axes.axvline(x = meanValue, color = meanLineColor, linewidth = 1.5)\n\n";
        ss << "axes.axvline(x = weightedMedianValue, color = weightedMedianLineColor, linewidth = 1.5)\n\n";
        ss << "axes.axvline(x = firstWeightedQuartileValue, color = quartileLineColor, linewidth = 1.5)\n\n";
        ss << "axes.axvline(x = thirdWeightedQuartileValue, color = quartileLineColor, linewidth = 1.5)\n\n";
        ss << "axes.axvline(x = minValue, color='black', linestyle = ':', linewidth = 1.5)\n\n";
        ss << "axes.axvline(x = maxValue, color='black', linestyle = ':', linewidth = 1.5)\n\n";

        ss << "if len(sys.argv) > 1:\n";
        ss << "\tif sys.argv[1] == '-save':\n";
        ss << "\t\tplt.savefig(sys.argv[2])\n";
        ss << "else:\n";
        ss << "\tplt.show()\n";

        ss << std::endl;

        return ss.str();
    }
};

class pooling
{
private:
    size_t mValueCount;
    double mValueSum;
    double mSquareValueSum;
    double mMinValue;
    double mMaxValue;
    double mVarianceBias = 0.0;
    uint32_t mMinCoord[2];
    uint32_t mMaxCoord[2];
    histogram mHistogram = histogram(100);
public:
    pooling() { clear(); }
    pooling(size_t buckets, double varianceBias = 0.0) : mVarianceBias(varianceBias) { mHistogram.resize(buckets); clear(); }
    
    histogram& getHistogram() { return mHistogram; }
    double getMinValue() const { return mMinValue; }
    double getMaxValue() const { return mMaxValue; }
    double getMean() const { return mValueSum / mValueCount; }
    double getVariance() const
    {
        bool isSampled = false;
        double variance = (mSquareValueSum - ((mValueSum * mValueSum) / mValueCount)) / (mValueCount - (isSampled ? 1 : 0));
        variance = std::max(0.0, variance);  //  negative not allowed, but can happen due to precision issues
        return variance;
    }

    double getStdDev(void) const { return std::sqrt(this->getVariance()); }

    double getCenterOfGravity(void) const 
    {
        double bucketStep = mHistogram.getBucketStep();
        double sumMoments = 0.0;
        double sumWeightedDataValue = 0.0;
        for (size_t bucketId = 0; bucketId < mHistogram.size(); bucketId++)
        {
            double weight = (bucketId + 0.5) * bucketStep;
            double weightedValue = mHistogram.getBucketValue(bucketId) * weight;
            sumWeightedDataValue += weightedValue;
            sumMoments += weightedValue * weight;
        }
        double cog = sumMoments / sumWeightedDataValue;
        return cog;
    }

    double getWeightedPercentile(const double percent) const  // percent = 0.5 gives you the weighted median. This function is approximative, since we do not have all the values. We find the bucket where the weighted median is located, and interpolate in there.
    {
        double weight;
        double weightedValue;
        double bucketStep = mHistogram.getBucketStep();
        double sumWeightedDataValue = 0.0;
        for (size_t bucketId = 0; bucketId < mHistogram.size(); bucketId++)
        {
            weight = (bucketId + 0.5) * bucketStep;
            weightedValue = mHistogram.getBucketValue(bucketId) * weight;
            sumWeightedDataValue += weightedValue;
        }

        double sum = 0;
        size_t weightedMedianIndex = 0;
        for (size_t bucketId = 0; bucketId < mHistogram.size(); bucketId++)
        {
            weight = (bucketId + 0.5) * bucketStep;
            weightedValue = mHistogram.getBucketValue(bucketId) * weight;
            weightedMedianIndex = bucketId;
            if (sum + weightedValue > percent * sumWeightedDataValue)
                break;
            sum += weightedValue;
        }

        weight = (weightedMedianIndex + 0.5) * bucketStep;
        weightedValue = mHistogram.getBucketValue(weightedMedianIndex) * weight;
        double discrepancy = percent * sumWeightedDataValue - sum;
        double linearWeight = discrepancy / weightedValue; // in [0,1]
        double percentile = (weightedMedianIndex + linearWeight) * bucketStep;
        return percentile;
    }

    double getPercentile(const double percent, size_t numCounts) const  // percent = 0.5 gives you the median. This function is approximative, since we do not have all the values. We find the bucket where the median is located and returns its weight.
    {
        size_t sum = 0;
        size_t medianIndex = 0;
        for (size_t bucketId = 0; bucketId < mHistogram.size(); bucketId++)
        {
            size_t count = mHistogram.getBucketValue(bucketId);
            medianIndex = bucketId;
            if (sum + count > size_t(percent * numCounts))
                break;
            sum += count;
        }
        double bucketStep = mHistogram.getBucketStep();
        double percentile = (medianIndex + 0.5) * bucketStep;
        return percentile;
    }

    void clear()
    {
        mValueCount = 0;
        mValueSum = 0.0;
        mSquareValueSum = 0.0;
        mMinValue = std::numeric_limits<double>::max();
        mMaxValue = std::numeric_limits<double>::min();
        mHistogram.clear();
    }

    void update(uint32_t xcoord, uint32_t ycoord, double value)
    {
        mValueCount++;
        mValueSum += value;
        mSquareValueSum += (value * value);
        mHistogram.inc(value);

        if (value < mMinValue)
        {
            mMinValue = value;
            mMinCoord[0] = xcoord;
            mMinCoord[1] = ycoord;
        }

        if (value > mMaxValue)
        {
            mMaxValue = value;
            mMaxCoord[0] = xcoord;
            mMaxCoord[1] = ycoord;
        }
    }

    std::string toString(bool verbose = false)
    {
        std::stringstream s;
        if (verbose)
        {
            s << std::fixed << std::setprecision(5) << "Mean = " << this->getMean() << ", Weighted median = " << this->getWeightedPercentile(0.5) << ", 1st weighted quartile = " << this->getWeightedPercentile(0.25) << ", 3rd weighted quartile = " << this->getWeightedPercentile(0.75) << ", Min value = " << mMinValue << " @ (" << mMinCoord[0] << "," << mMinCoord[1] << "), Max value =  " << mMaxValue << " @ (" << mMaxCoord[0] << ", " << mMaxCoord[1] << ")\n";
        }
        else
        {
            s << "Mean ; Weighted median ; 1st weighted quartile; 3rd weighted quartile ; Min value ; MinPosX; MinPosY; Max value; MaxPosX; MaxPosY\n";
            s << std::fixed << std::setprecision(5) << this->getMean() << ";" << this->getWeightedPercentile(0.5) << ";" << this->getWeightedPercentile(0.25) << ";" << this->getWeightedPercentile(0.75) << ";" << mMinValue << ";" << mMinCoord[0] << ";" << mMinCoord[1] << ";" << mMaxValue << ";" << mMaxCoord[0] << ";" << mMaxCoord[1] << "\n";
        }
        return s.str();
    }

    void save(const std::string fileName, float ppd, int imgWidth, int imgHeight, const bool verbose, const bool optionLog, const std::string referenceFileName, const std::string testFileName)
    {
        std::ofstream file;
        if (verbose)
        {
            std::cout << "Writing metric histogram to file <" << fileName << ".csv> and <" << fileName << ".py> with image resolution: " << imgWidth << "x" << imgHeight << "\n";
        }

        file.open(fileName + ".csv");
        file << "#  FLIP pooling statistics for <" << referenceFileName << "> vs. <" << testFileName << ">\n";

        file << this->toString(false);

        file << "#  histogram\n";
        file << "Min bucket" << ";" << "Max bucket" << "\n";
        file << mHistogram.getBucketIdMin() << ";" << mHistogram.getBucketIdMax() << "\n";
        file << "Bucket no; ";
        for (size_t bucketId = 0; bucketId < mHistogram.size(); bucketId++)
        {
            file << bucketId << ";";
        }
        file << "\n";

        if (verbose)
        {
            std::cout << this->toString(true);
            std::cout << "# Histogram bucket size = " << mHistogram.getBucketSize() << ", range = [" << mHistogram.getBucketIdMin() << ", " << mHistogram.getBucketIdMax() << "]\n";
            std::cout << "Non-weighted bucket counts:\n";
        }

        size_t area = size_t(imgWidth) * size_t(imgHeight);

        file << "Count; ";
        for (size_t bucketId = 0; bucketId < mHistogram.size(); bucketId++)
        {
            size_t histogramValue = mHistogram.getBucketValue(bucketId);
            if (verbose)
            {
                std::cout << bucketId << ": " << histogramValue << "\n";
            }
            file << float(histogramValue) << ";";
        }
        file << "\n";

        file << "Weight; ";
        double bucketStep = mHistogram.getBucketStep();
        for (size_t bucketId = 0; bucketId < mHistogram.size(); bucketId++)
        {
            file << (0.5 + bucketId) * bucketStep << ";";
        }
        file << "\n";

        file << "Weighted bucket count per megapixel; ";
        double megaPixelWeight = 1024.0 * 1024.0 / double(area);
        for (size_t bucketId = 0; bucketId < mHistogram.size(); bucketId++)
        {
            file << ((0.5 + bucketId) * bucketStep * mHistogram.getBucketValue(bucketId) * megaPixelWeight) << ";";
        }
        file << "\n";

        //  python output
        std::ofstream pythonHistogramFile;
        pythonHistogramFile.open(fileName + ".py");
        pythonHistogramFile << mHistogram.toPython(area, ppd, getMean(), getStdDev(), getMaxValue(), getMinValue(), getWeightedPercentile(0.5), getWeightedPercentile(0.25), getWeightedPercentile(0.75),  optionLog);
        pythonHistogramFile.close();
    }
};

} // namespace flip_detail

// flip.cpp

#include <algorithm>

namespace flip_detail {

float gPPD;
float gMonitorDistance = 0.7f; 
float gMonitorWidth = 0.7f;
float gMonitorResolutionX = 3840.0f;
const float gqc = 0.7f;
const float gpc = 0.4f;
const float gpt = 0.95f;
const float gw = 0.082f;
const float gqf = 0.5f;

//  Pixels per degree (PPD)
static float calculatePPD(const float dist, const float resolutionX, const float monitorWidth)
{
    return dist * (resolutionX / monitorWidth) * ( float(M_PI) / 180.0f );
}

class color3
{
public:
    color3() { x = 0.0f; y = 0.0f; z = 0.0f; }
    color3(const float _x, const float _y, const float _z) { x = _x; y = _y; z = _z; }
    color3& abs() { x = fabsf(x); y = fabsf(y); z = fabsf(z); return *this; }
    color3& max(const float value) { x = std::max(x, value); y = std::max(y, value); z = std::max(z, value); return *this; }
    color3& min(const float value) { x = std::min(x, value); y = std::min(y, value); z = std::min(z, value); return *this; }
    color3& min(const color3& value) { x = std::min(x, value.x); y = std::min(y, value.y);  z = std::min(z, value.z); return *this; }
    color3& sRGB2LinearRGB();
    color3& LinearRGB2sRGB();
    color3& LinearRGB2XYZ();
    color3& XYZ2LinearRGB();
    color3& XYZ2YCxCz(const color3 reference_illuminant = color3(0.950428545377181f, 1.0f, 1.088900370798128f));      // the default illuminant is D65
    color3& YCxCz2XYZ(const color3 reference_illuminant = color3(0.950428545377181f, 1.0f, 1.088900370798128f));      // the default illuminant is D65
    color3& XYZ2CIELab(const color3 reference_illuminant = color3(0.950428545377181f, 1.0f, 1.088900370798128f));     // the default illuminant is D65
    color3& CIELab2XYZ(const color3 reference_illuminant = color3(0.950428545377181f, 1.0f, 1.088900370798128f));     // the default illuminant is D65
public:
    float x, y, z;
};

static float sRGB2Linear(float sRGBColor)
{
    if (sRGBColor <= 0.04045f)
        return sRGBColor / 12.92f;
    else
        return powf((sRGBColor + 0.055f) / 1.055f, 2.4f);
}

static float Linear2sRGB(float linearColor)
{
    if (linearColor <= 0.0031308f)
        return linearColor * 12.92f;
    else
        return 1.055f * powf(linearColor, 1.0f / 2.4f) - 0.055f;
}

color3& color3::sRGB2LinearRGB()
{
    x = sRGB2Linear(x);
    y = sRGB2Linear(y);
    z = sRGB2Linear(z);
    return *this;
}

color3& color3::LinearRGB2sRGB()
{
    x = Linear2sRGB(x);
    y = Linear2sRGB(y);
    z = Linear2sRGB(z);
    return *this;
}

color3& color3::LinearRGB2XYZ()    // Source: https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
{                               // Assumes D65 standard illuminant
    const float a11 = 10135552.0f / 24577794.0f;
    const float a12 = 8788810.0f / 24577794.0f;
    const float a13 = 4435075.0f / 24577794.0f;
    const float a21 = 2613072.0f / 12288897.0f;
    const float a22 = 8788810.0f / 12288897.0f;
    const float a23 = 887015.0f / 12288897.0f;
    const float a31 = 1425312.0f / 73733382.0f;
    const float a32 = 8788810.0f / 73733382.0f;
    const float a33 = 70074185.0f / 73733382.0f;
    color3 v = *this;
    x = a11 * v.x + a12 * v.y + a13 * v.z;
    y = a21 * v.x + a22 * v.y + a23 * v.z;
    z = a31 * v.x + a32 * v.y + a33 * v.z;
    return *this;
}

color3& color3::XYZ2LinearRGB()    // Return values in linear RGB, assuming D65 standard illuminant
{
    const float a11 = 3.241003232976358f;
    const float a12 = -1.537398969488785f;
    const float a13 = -0.498615881996363f;
    const float a21 = -0.969224252202516f;
    const float a22 = 1.875929983695176f;
    const float a23 = 0.041554226340085f;
    const float a31 = 0.055639419851975f;
    const float a32 = -0.204011206123910f;
    const float a33 = 1.057148977187533f;
    color3 v = *this;
    x = a11 * v.x + a12 * v.y + a13 * v.z;
    y = a21 * v.x + a22 * v.y + a23 * v.z;
    z = a31 * v.x + a32 * v.y + a33 * v.z;
    return *this;
}

color3& color3::XYZ2YCxCz(const color3 reference_illuminant)     // the default illuminant is D65
{
    color3 xyz = *this;
    xyz.x /= reference_illuminant.x;
    xyz.y /= reference_illuminant.y;
    xyz.z /= reference_illuminant.z;
    const float Y = 116.0f * xyz.y - 16.0f;
    const float Cx = 500.0f * (xyz.x - xyz.y);
    const float Cz = 200.0f * (xyz.y - xyz.z);
    x = Y;
    y = Cx;
    z = Cz;
    return *this;
}

color3& color3::YCxCz2XYZ(const color3 reference_illuminant)     // the default illuminant is D65
{
    color3 YCxCz = *this;
    const float Yy = (YCxCz.x + 16.0f) / 116.0f;
    const float Cx = YCxCz.y / 500.0f;
    const float Cz = YCxCz.z / 200.0f;
    x = Yy + Cx;
    y = Yy;
    z = Yy - Cz;
    x *= reference_illuminant.x;
    y *= reference_illuminant.y;
    z *= reference_illuminant.z;
    return *this;
}

color3& color3::XYZ2CIELab(const color3 reference_illuminant)     // the default illuminant is D65
{
    color3 xyz = *this;
    xyz.abs();
    xyz.x /= reference_illuminant.x;
    xyz.y /= reference_illuminant.y;
    xyz.z /= reference_illuminant.z;
    xyz.x = xyz.x > 0.008856 ? powf(xyz.x, 1.0f / 3.0f) : 7.787f * xyz.x + 16.0f / 116.0f;
    xyz.y = xyz.y > 0.008856 ? powf(xyz.y, 1.0f / 3.0f) : 7.787f * xyz.y + 16.0f / 116.0f;
    xyz.z = xyz.z > 0.008856 ? powf(xyz.z, 1.0f / 3.0f) : 7.787f * xyz.z + 16.0f / 116.0f;
    x = 116.0f * xyz.y - 16.0f;
    y = 500.0f * (xyz.x - xyz.y);
    z = 200.0f * (xyz.y - xyz.z);
    return *this;
}

color3& color3::CIELab2XYZ(const color3 reference_illuminant)     // the default illuminant is D65
{
    color3 LAB = *this;
    float Y = (LAB.x + 16.0f) / 116.0f;
    float X = LAB.y / 500.0f + Y;
    float Z = Y - LAB.z / 200.0f;
    X = ((X > 0.206897) ? X * X * X : (X - 16.0f / 116.0f) / 7.787f);
    Y = ((Y > 0.206897) ? Y * Y * Y : (Y - 16.0f / 116.0f) / 7.787f);
    Z = ((Z > 0.206897) ? Z * Z * Z : (Z - 16.0f / 116.0f) / 7.787f);
    x = X * reference_illuminant.x;
    y = Y * reference_illuminant.y;
    z = Z * reference_illuminant.z;
    return *this;
}

static float HyAB(color3& refPixel, color3& testPixel)
{
    float cityBlockDistanceL = fabsf(refPixel.x - testPixel.x);
    float euclideanDistanceAB = sqrtf((refPixel.y - testPixel.y) * (refPixel.y - testPixel.y) + (refPixel.z - testPixel.z) * (refPixel.z - testPixel.z));
    return cityBlockDistanceL + euclideanDistanceAB;
}

static float Gaussian(const float x, const float y, const float sigma)
{
    return expf(-(x * x + y * y) / (2.0f * sigma * sigma));
}

class image
{
public:
    image()             { mWidth = 0; mHeight = 0; }
    image(const int width, const int height) { allocate(width, height); }
    void                allocate(const int width, const int height) { mWidth = width; mHeight = height; mPixels.resize(mWidth * mHeight, color3(0.0f, 0.0f, 0.0f)); }
    void                preprocess(image& filter);
    void                convolve(image& unfilteredImage, image& filter);
    void                huntAdjustment();
    void                generateSpatialFilter(const float ppd);
    void                generateDetectionFilters(const float ppd, const bool pointDetector);
    color3&             get(const int x, const int y) { return mPixels[y * mWidth + x]; }
    void                set(const int x, const int y, const color3& color) { mPixels[y * mWidth + x] = color; }
    void                computeColorDifference(image& preprocessedReference, image& preprocessedTest);
    void                computeFeatureDifference(image& refImage, image& testImage, const float ppd);
    void                computeFLIPError(image& testImage, image& refImage, bool verbose);
    void                remapToMagma();
public:
    int                 mWidth = 0;
    int                 mHeight = 0;
    std::vector<color3> mPixels;
};

static float GaussSum(const float x2, const float a1, const float b1, const float a2, const float b2)
{
    const float pi = float(M_PI);
    const float pi_sq = float(M_PI * M_PI);
    return a1 * sqrtf(pi / b1) * expf(-pi_sq * x2 / b1) + a2 * sqrtf(pi / b2) * expf(-pi_sq * x2 / b2);
}

void image::generateSpatialFilter(const float ppd)
{
    const float deltaX = 1.0f / ppd;
    const float pi_sq = float(M_PI * M_PI);
    // constants for Gaussians -- see paper for details.
    color3 a1 = {1.0f, 1.0f, 34.1f};
    color3 b1 = {0.0047f, 0.0053f, 0.04f };
    color3 a2 = { 0.0f, 0.0f, 13.5f };
    color3 b2 = { 1.0e-5f, 1.0e-5f, 0.025f };

    float maxScaleParameter = std::max(std::max(std::max(b1.x, b1.y), std::max(b1.z, b2.x)), std::max(b2.y, b2.z));
    int radius = int(std::ceil(3.0f * sqrtf(maxScaleParameter / (2.0f * pi_sq)) * ppd)); // Set radius based on largest scale parameter

    int width = 2 * radius + 1;
    allocate(width, width);

    color3 filterSum = { 0.0f, 0.0f, 0.0f };
    for (int y = 0; y < width; y++)
    {
        float iy = (y - radius) * deltaX;
        for (int x = 0; x < width; x++)
        {   
            float ix = (x - radius) * deltaX;

            float dist2 = ix * ix + iy * iy;
            color3 value = color3(GaussSum(dist2, a1.x, b1.x, a2.x, b2.x), GaussSum(dist2, a1.y, b1.y, a2.y, b2.y), GaussSum(dist2, a1.z, b1.z, a2.z, b2.z));
            this->set(x, y, value);
            filterSum = color3(filterSum.x + value.x, filterSum.y + value.y, filterSum.z + value.z);
        }
    }

    // normalize weights
    for (int y = 0; y < width; y++)
    {
        for (int x = 0; x < width; x++)
        {
            color3 value = this->get(x, y);
            this->set(x, y, color3(value.x / filterSum.x, value.y / filterSum.y, value.z / filterSum.z));
        }
    }
}

void image::convolve(image& unfilteredImage, image& filter)     // assumes a filter whose weights sum to one, and we use "replicate" for borders, i.e., clamp to borders
{
    assert(filter.mWidth % 2 == 1 && filter.mHeight % 2 == 1);  // need to have odd-sized filter
    const int halfFilterWidth = filter.mWidth / 2;
    const int halfFilterHeight = filter.mHeight / 2;

    // apply filter
    for (int y = 0; y < mHeight; y++)
    {
        for (int x = 0; x < mWidth; x++)
        {
            color3 colorSum = color3(0.0f, 0.0f, 0.0f);

            // filter with the kernel
            for (int iy = -halfFilterHeight; iy <= halfFilterHeight; iy++)
            {
                int yy = std::min(std::max(0, y + iy), mHeight - 1);
                for (int ix = -halfFilterWidth; ix <= halfFilterWidth; ix++)
                {
                    int xx = std::min(std::max(0, x + ix), mWidth - 1);
                    // read from unfilteredImage and sum into colorSum and weightSum
                    const color3 weights = filter.get(ix + halfFilterWidth, iy + halfFilterHeight);
                    color3 srcColor = unfilteredImage.get(xx, yy);
                    colorSum = color3(colorSum.x + weights.x * srcColor.x, colorSum.y + weights.y * srcColor.y, colorSum.z + weights.z * srcColor.z);
                }
            }
            this->set(x, y, color3(colorSum.x, colorSum.y, colorSum.z));     // compute final filtered value (and store in mPixels)
        }
    }
}

static float Hunt(const float luminance, const float chrominance)
{
    return 0.01f * luminance * chrominance;
}

void image::huntAdjustment()
{
    for (auto& pixel : mPixels)
    {
        pixel.y = Hunt(pixel.x, pixel.y);
        pixel.z = Hunt(pixel.x, pixel.z);
    }
}

void image::preprocess(image& filter)     // implements spatial filter and Hunt adjustment
{
    // convolve with CSF filters
    image inputImage = *this;              // copy the input image
    this->convolve(inputImage, filter);

    // the result after the spatial filter is in YCxCz. Clamp and transform this->mPixels to L*a*b
    for (auto& pixel : this->mPixels)
        pixel.YCxCz2XYZ().XYZ2LinearRGB().min(1.0f).max(0.0f).LinearRGB2XYZ().XYZ2CIELab();

    huntAdjustment();
}

static float computeMaxDistance()
{
    color3 greenLab = color3(0.0f, 1.0f, 0.0f);     // in RGB to start with
    color3 blueLab = color3(0.0f, 0.0f, 1.0f);
    greenLab.LinearRGB2XYZ().XYZ2CIELab();          // now in Lab
    blueLab.LinearRGB2XYZ().XYZ2CIELab();
    color3 greenLabHunt = color3(greenLab.x, Hunt(greenLab.x, greenLab.y), Hunt(greenLab.x, greenLab.z));
    color3 blueLabHunt = color3(blueLab.x, Hunt(blueLab.x, blueLab.y), Hunt(blueLab.x, blueLab.z));

    return powf(HyAB(greenLabHunt, blueLabHunt), gqc);
}

void image::computeColorDifference(image& preprocessedReference, image& preprocessedTest)
{
    this->allocate(preprocessedReference.mWidth, preprocessedReference.mHeight);

    const float cmax = computeMaxDistance();
    const float pccmax = gpc * cmax;

    // compute difference in HyAB and redistribute errors
    for (int y = 0; y < this->mHeight; y++)
    {
        for (int x = 0; x < this->mWidth; x++)
        {
            // compute difference in HyAB
            color3 refPixel = preprocessedReference.get(x, y);
            color3 testPixel = preprocessedTest.get(x, y);
            float error = HyAB(refPixel, testPixel);

            error = powf(error, gqc);

            // Re-map error to the [0, 1] range. Values between 0 and pccmax are mapped to the range [0, gpt],
            // while the rest are mapped to the range (gpt, 1]
            if (error < pccmax)
            {
                error *= gpt / pccmax;
            }
            else
            {
                error = gpt + ((error - pccmax) / (cmax - pccmax)) * (1.0f - gpt);
            }

            this->set(x, y, color3(error, 0.0f, 0.0f));
        }
    }
}

void image::generateDetectionFilters(const float ppd, const bool pointDetector)
{
    const float stdDev = 0.5f * gw * ppd;
    const int radius = int(std::ceil(3.0f * stdDev));
    const int width = 2 * radius + 1;

    this->allocate(width, width);

    float weightX, weightY;
    float negativeWeightsSumX = 0.0f;
    float positiveWeightsSumX = 0.0f;
    float negativeWeightsSumY = 0.0f;
    float positiveWeightsSumY = 0.0f;

    for (int y = 0; y < width; y++)
    {
        int yy = y - radius;
        for (int x = 0; x < width; x++)
        {
            int xx = x - radius;
            float G = Gaussian(float(xx), float(yy), stdDev);
            if (pointDetector)
            {
                weightX = (float(xx) * float(xx) / (stdDev * stdDev) - 1.0f) * G;
                weightY = (float(yy) * float(yy) / (stdDev * stdDev) - 1.0f) * G;
            }
            else
            {
                weightX = -float(xx) * G;
                weightY = -float(yy) * G;
            }
            this->set(x, y, color3(weightX, weightY, 0.0f));

            if (weightX > 0.0f)
                positiveWeightsSumX += weightX;
            else
                negativeWeightsSumX += -weightX;

            if (weightY > 0.0f)
                positiveWeightsSumY += weightY;
            else
                negativeWeightsSumY += -weightY;
        }
    }

    // Normalize positive weights to sum to 1 and negative weights to sum to -1
    for (int y = 0; y < width; y++)
    {
        for (int x = 0; x < width; x++)
        {
            color3 p = this->get(x, y);

            this->set(x, y, color3(p.x / (p.x > 0.0f ? positiveWeightsSumX : negativeWeightsSumX), p.y / (p.y > 0.0f ? positiveWeightsSumY : negativeWeightsSumY), 0.0f));
        }
    }
}

void image::computeFeatureDifference(image& refImage, image& testImage, const float ppd)
{
    this->allocate(refImage.mWidth, testImage.mHeight);

    // make copy of ref and test images and extract and normalize achromatic component
    image refGray = refImage;
    image testGray = testImage;
    for (auto& pixel : refGray.mPixels)
    {
        float c = (pixel.x + 16.0f) / 116.0f;   // make it [0,1]
        pixel = color3(c, c, 0.0f);             // luminance [0,1] stored in both x and y since we apply both horizontal and verticals filters at the same time
    }

    for (auto& pixel : testGray.mPixels)
    {
        float c = (pixel.x + 16.0f) / 116.0f;   // make it [0,1]
        pixel = color3(c, c, 0.0f);             // luminance [0,1] stored in both x and y since we apply both horizontal and verticals filters at the same time
    }

    image edgesReference(this->mWidth, this->mHeight);
    image pointsReference(this->mWidth, this->mHeight);
    image edgesTest(this->mWidth, this->mHeight);
    image pointsTest(this->mWidth, this->mHeight);

    image edgeFilters;
    image pointFilters;
    edgeFilters.generateDetectionFilters(ppd, false); // contains both horiontal and vertical filter
    pointFilters.generateDetectionFilters(ppd, true); // contains both horiontal and vertical filter

    edgesReference.convolve(refGray, edgeFilters);
    edgesTest.convolve(testGray, edgeFilters);
    pointsReference.convolve(refGray, pointFilters);
    pointsTest.convolve(testGray, pointFilters);

    const float normalizationFactor = 1.0f / sqrtf(2.0f);
    color3 p;
    for (int y = 0; y < this->mHeight; y++)
    {
        for (int x = 0; x < this->mWidth; x++)
        {
            p = edgesReference.get(x, y);
            const float edgeValueRef = sqrtf(p.x * p.x + p.y * p.y);
            p = edgesTest.get(x, y);
            const float edgeValueTest = sqrtf(p.x * p.x + p.y * p.y);
            p = pointsReference.get(x, y);
            const float pointValueRef = sqrtf(p.x * p.x + p.y * p.y);
            p = pointsTest.get(x, y);
            const float pointValueTest = sqrtf(p.x * p.x + p.y * p.y);

            const float edgeDifference = std::abs(edgeValueRef - edgeValueTest);
            const float pointDifference = std::abs(pointValueRef - pointValueTest);

            const float featureDifference = std::pow(normalizationFactor * std::max(edgeDifference, pointDifference), gqf);

            this->set(x, y, color3(featureDifference, 0.0f, 0.0f));
        }
    }
}

void image::computeFLIPError(image& refImage, image& testImage, bool verbose)
{
    // Transform refImage and testImage to YCxCz opponent space
    for (auto& pixel : refImage.mPixels)
    {
        pixel.sRGB2LinearRGB().LinearRGB2XYZ().XYZ2YCxCz();
    }
    for (auto& pixel : testImage.mPixels)
    {
        pixel.sRGB2LinearRGB().LinearRGB2XYZ().XYZ2YCxCz();
    }

    // Color pipeline
    // Preprocess images
    image preprocessedReference = refImage;
    image preprocessedTest = testImage;
    image filter;
    filter.generateSpatialFilter(gPPD);
    if (verbose) std::cout << "Preprocessing reference image..." << std::flush;
    preprocessedReference.preprocess(filter);
    if (verbose) std::cout << "ok.\nPreprocessing test image..." << std::flush;
    preprocessedTest.preprocess(filter);
    if (verbose) std::cout << "ok.\nComputing error map..." << std::flush;
    // Compute color difference
    image colorDifference;
    colorDifference.computeColorDifference(preprocessedReference, preprocessedTest);

    // Feature pipeline
    image featureDifference;
    featureDifference.computeFeatureDifference(refImage, testImage, gPPD);

    // Final error
    this->allocate(colorDifference.mWidth, colorDifference.mHeight);
    for (int y = 0; y < this->mHeight; y++)
    {
        for (int x = 0; x < this->mWidth; x++)
        {
            const float cdiff = colorDifference.get(x, y).x;
            const float fdiff = featureDifference.get(x, y).x;
            const float errorFLIP = std::pow(cdiff, 1.0f - fdiff);
            this->set(x, y, color3(errorFLIP, errorFLIP, errorFLIP));
        }
    }
}

//pooling& handlePooling(image& img, float ppd, const std::string fileName, const bool verbose, const bool optionLog, const std::string referenceFileName, const std::string testFileName)
void handlePooling(image& image, pooling& pooling)
{
    // add all FLIP values in the error map to pooling
    for (int y = 0; y < image.mHeight; y++)
    {
        for (int x = 0; x < image.mWidth; x++)
        {
            float value = image.get(x, y).x;
            pooling.update(x, y, value);
        }
    }
}

static const color3 magmaMap[256] = {
    {0.001462f, 0.000466f, 0.013866f}, {0.002258f, 0.001295f, 0.018331f}, {0.003279f, 0.002305f, 0.023708f}, {0.004512f, 0.003490f, 0.029965f},
    {0.005950f, 0.004843f, 0.037130f}, {0.007588f, 0.006356f, 0.044973f}, {0.009426f, 0.008022f, 0.052844f}, {0.011465f, 0.009828f, 0.060750f},
    {0.013708f, 0.011771f, 0.068667f}, {0.016156f, 0.013840f, 0.076603f}, {0.018815f, 0.016026f, 0.084584f}, {0.021692f, 0.018320f, 0.092610f},
    {0.024792f, 0.020715f, 0.100676f}, {0.028123f, 0.023201f, 0.108787f}, {0.031696f, 0.025765f, 0.116965f}, {0.035520f, 0.028397f, 0.125209f},
    {0.039608f, 0.031090f, 0.133515f}, {0.043830f, 0.033830f, 0.141886f}, {0.048062f, 0.036607f, 0.150327f}, {0.052320f, 0.039407f, 0.158841f},
    {0.056615f, 0.042160f, 0.167446f}, {0.060949f, 0.044794f, 0.176129f}, {0.065330f, 0.047318f, 0.184892f}, {0.069764f, 0.049726f, 0.193735f},
    {0.074257f, 0.052017f, 0.202660f}, {0.078815f, 0.054184f, 0.211667f}, {0.083446f, 0.056225f, 0.220755f}, {0.088155f, 0.058133f, 0.229922f},
    {0.092949f, 0.059904f, 0.239164f}, {0.097833f, 0.061531f, 0.248477f}, {0.102815f, 0.063010f, 0.257854f}, {0.107899f, 0.064335f, 0.267289f},
    {0.113094f, 0.065492f, 0.276784f}, {0.118405f, 0.066479f, 0.286321f}, {0.123833f, 0.067295f, 0.295879f}, {0.129380f, 0.067935f, 0.305443f},
    {0.135053f, 0.068391f, 0.315000f}, {0.140858f, 0.068654f, 0.324538f}, {0.146785f, 0.068738f, 0.334011f}, {0.152839f, 0.068637f, 0.343404f},
    {0.159018f, 0.068354f, 0.352688f}, {0.165308f, 0.067911f, 0.361816f}, {0.171713f, 0.067305f, 0.370771f}, {0.178212f, 0.066576f, 0.379497f},
    {0.184801f, 0.065732f, 0.387973f}, {0.191460f, 0.064818f, 0.396152f}, {0.198177f, 0.063862f, 0.404009f}, {0.204935f, 0.062907f, 0.411514f},
    {0.211718f, 0.061992f, 0.418647f}, {0.218512f, 0.061158f, 0.425392f}, {0.225302f, 0.060445f, 0.431742f}, {0.232077f, 0.059889f, 0.437695f},
    {0.238826f, 0.059517f, 0.443256f}, {0.245543f, 0.059352f, 0.448436f}, {0.252220f, 0.059415f, 0.453248f}, {0.258857f, 0.059706f, 0.457710f},
    {0.265447f, 0.060237f, 0.461840f}, {0.271994f, 0.060994f, 0.465660f}, {0.278493f, 0.061978f, 0.469190f}, {0.284951f, 0.063168f, 0.472451f},
    {0.291366f, 0.064553f, 0.475462f}, {0.297740f, 0.066117f, 0.478243f}, {0.304081f, 0.067835f, 0.480812f}, {0.310382f, 0.069702f, 0.483186f},
    {0.316654f, 0.071690f, 0.485380f}, {0.322899f, 0.073782f, 0.487408f}, {0.329114f, 0.075972f, 0.489287f}, {0.335308f, 0.078236f, 0.491024f},
    {0.341482f, 0.080564f, 0.492631f}, {0.347636f, 0.082946f, 0.494121f}, {0.353773f, 0.085373f, 0.495501f}, {0.359898f, 0.087831f, 0.496778f},
    {0.366012f, 0.090314f, 0.497960f}, {0.372116f, 0.092816f, 0.499053f}, {0.378211f, 0.095332f, 0.500067f}, {0.384299f, 0.097855f, 0.501002f},
    {0.390384f, 0.100379f, 0.501864f}, {0.396467f, 0.102902f, 0.502658f}, {0.402548f, 0.105420f, 0.503386f}, {0.408629f, 0.107930f, 0.504052f},
    {0.414709f, 0.110431f, 0.504662f}, {0.420791f, 0.112920f, 0.505215f}, {0.426877f, 0.115395f, 0.505714f}, {0.432967f, 0.117855f, 0.506160f},
    {0.439062f, 0.120298f, 0.506555f}, {0.445163f, 0.122724f, 0.506901f}, {0.451271f, 0.125132f, 0.507198f}, {0.457386f, 0.127522f, 0.507448f},
    {0.463508f, 0.129893f, 0.507652f}, {0.469640f, 0.132245f, 0.507809f}, {0.475780f, 0.134577f, 0.507921f}, {0.481929f, 0.136891f, 0.507989f},
    {0.488088f, 0.139186f, 0.508011f}, {0.494258f, 0.141462f, 0.507988f}, {0.500438f, 0.143719f, 0.507920f}, {0.506629f, 0.145958f, 0.507806f},
    {0.512831f, 0.148179f, 0.507648f}, {0.519045f, 0.150383f, 0.507443f}, {0.525270f, 0.152569f, 0.507192f}, {0.531507f, 0.154739f, 0.506895f},
    {0.537755f, 0.156894f, 0.506551f}, {0.544015f, 0.159033f, 0.506159f}, {0.550287f, 0.161158f, 0.505719f}, {0.556571f, 0.163269f, 0.505230f},
    {0.562866f, 0.165368f, 0.504692f}, {0.569172f, 0.167454f, 0.504105f}, {0.575490f, 0.169530f, 0.503466f}, {0.581819f, 0.171596f, 0.502777f},
    {0.588158f, 0.173652f, 0.502035f}, {0.594508f, 0.175701f, 0.501241f}, {0.600868f, 0.177743f, 0.500394f}, {0.607238f, 0.179779f, 0.499492f},
    {0.613617f, 0.181811f, 0.498536f}, {0.620005f, 0.183840f, 0.497524f}, {0.626401f, 0.185867f, 0.496456f}, {0.632805f, 0.187893f, 0.495332f},
    {0.639216f, 0.189921f, 0.494150f}, {0.645633f, 0.191952f, 0.492910f}, {0.652056f, 0.193986f, 0.491611f}, {0.658483f, 0.196027f, 0.490253f},
    {0.664915f, 0.198075f, 0.488836f}, {0.671349f, 0.200133f, 0.487358f}, {0.677786f, 0.202203f, 0.485819f}, {0.684224f, 0.204286f, 0.484219f},
    {0.690661f, 0.206384f, 0.482558f}, {0.697098f, 0.208501f, 0.480835f}, {0.703532f, 0.210638f, 0.479049f}, {0.709962f, 0.212797f, 0.477201f},
    {0.716387f, 0.214982f, 0.475290f}, {0.722805f, 0.217194f, 0.473316f}, {0.729216f, 0.219437f, 0.471279f}, {0.735616f, 0.221713f, 0.469180f},
    {0.742004f, 0.224025f, 0.467018f}, {0.748378f, 0.226377f, 0.464794f}, {0.754737f, 0.228772f, 0.462509f}, {0.761077f, 0.231214f, 0.460162f},
    {0.767398f, 0.233705f, 0.457755f}, {0.773695f, 0.236249f, 0.455289f}, {0.779968f, 0.238851f, 0.452765f}, {0.786212f, 0.241514f, 0.450184f},
    {0.792427f, 0.244242f, 0.447543f}, {0.798608f, 0.247040f, 0.444848f}, {0.804752f, 0.249911f, 0.442102f}, {0.810855f, 0.252861f, 0.439305f},
    {0.816914f, 0.255895f, 0.436461f}, {0.822926f, 0.259016f, 0.433573f}, {0.828886f, 0.262229f, 0.430644f}, {0.834791f, 0.265540f, 0.427671f},
    {0.840636f, 0.268953f, 0.424666f}, {0.846416f, 0.272473f, 0.421631f}, {0.852126f, 0.276106f, 0.418573f}, {0.857763f, 0.279857f, 0.415496f},
    {0.863320f, 0.283729f, 0.412403f}, {0.868793f, 0.287728f, 0.409303f}, {0.874176f, 0.291859f, 0.406205f}, {0.879464f, 0.296125f, 0.403118f},
    {0.884651f, 0.300530f, 0.400047f}, {0.889731f, 0.305079f, 0.397002f}, {0.894700f, 0.309773f, 0.393995f}, {0.899552f, 0.314616f, 0.391037f},
    {0.904281f, 0.319610f, 0.388137f}, {0.908884f, 0.324755f, 0.385308f}, {0.913354f, 0.330052f, 0.382563f}, {0.917689f, 0.335500f, 0.379915f},
    {0.921884f, 0.341098f, 0.377376f}, {0.925937f, 0.346844f, 0.374959f}, {0.929845f, 0.352734f, 0.372677f}, {0.933606f, 0.358764f, 0.370541f},
    {0.937221f, 0.364929f, 0.368567f}, {0.940687f, 0.371224f, 0.366762f}, {0.944006f, 0.377643f, 0.365136f}, {0.947180f, 0.384178f, 0.363701f},
    {0.950210f, 0.390820f, 0.362468f}, {0.953099f, 0.397563f, 0.361438f}, {0.955849f, 0.404400f, 0.360619f}, {0.958464f, 0.411324f, 0.360014f},
    {0.960949f, 0.418323f, 0.359630f}, {0.963310f, 0.425390f, 0.359469f}, {0.965549f, 0.432519f, 0.359529f}, {0.967671f, 0.439703f, 0.359810f},
    {0.969680f, 0.446936f, 0.360311f}, {0.971582f, 0.454210f, 0.361030f}, {0.973381f, 0.461520f, 0.361965f}, {0.975082f, 0.468861f, 0.363111f},
    {0.976690f, 0.476226f, 0.364466f}, {0.978210f, 0.483612f, 0.366025f}, {0.979645f, 0.491014f, 0.367783f}, {0.981000f, 0.498428f, 0.369734f},
    {0.982279f, 0.505851f, 0.371874f}, {0.983485f, 0.513280f, 0.374198f}, {0.984622f, 0.520713f, 0.376698f}, {0.985693f, 0.528148f, 0.379371f},
    {0.986700f, 0.535582f, 0.382210f}, {0.987646f, 0.543015f, 0.385210f}, {0.988533f, 0.550446f, 0.388365f}, {0.989363f, 0.557873f, 0.391671f},
    {0.990138f, 0.565296f, 0.395122f}, {0.990871f, 0.572706f, 0.398714f}, {0.991558f, 0.580107f, 0.402441f}, {0.992196f, 0.587502f, 0.406299f},
    {0.992785f, 0.594891f, 0.410283f}, {0.993326f, 0.602275f, 0.414390f}, {0.993834f, 0.609644f, 0.418613f}, {0.994309f, 0.616999f, 0.422950f},
    {0.994738f, 0.624350f, 0.427397f}, {0.995122f, 0.631696f, 0.431951f}, {0.995480f, 0.639027f, 0.436607f}, {0.995810f, 0.646344f, 0.441361f},
    {0.996096f, 0.653659f, 0.446213f}, {0.996341f, 0.660969f, 0.451160f}, {0.996580f, 0.668256f, 0.456192f}, {0.996775f, 0.675541f, 0.461314f},
    {0.996925f, 0.682828f, 0.466526f}, {0.997077f, 0.690088f, 0.471811f}, {0.997186f, 0.697349f, 0.477182f}, {0.997254f, 0.704611f, 0.482635f},
    {0.997325f, 0.711848f, 0.488154f}, {0.997351f, 0.719089f, 0.493755f}, {0.997351f, 0.726324f, 0.499428f}, {0.997341f, 0.733545f, 0.505167f},
    {0.997285f, 0.740772f, 0.510983f}, {0.997228f, 0.747981f, 0.516859f}, {0.997138f, 0.755190f, 0.522806f}, {0.997019f, 0.762398f, 0.528821f},
    {0.996898f, 0.769591f, 0.534892f}, {0.996727f, 0.776795f, 0.541039f}, {0.996571f, 0.783977f, 0.547233f}, {0.996369f, 0.791167f, 0.553499f},
    {0.996162f, 0.798348f, 0.559820f}, {0.995932f, 0.805527f, 0.566202f}, {0.995680f, 0.812706f, 0.572645f}, {0.995424f, 0.819875f, 0.579140f},
    {0.995131f, 0.827052f, 0.585701f}, {0.994851f, 0.834213f, 0.592307f}, {0.994524f, 0.841387f, 0.598983f}, {0.994222f, 0.848540f, 0.605696f},
    {0.993866f, 0.855711f, 0.612482f}, {0.993545f, 0.862859f, 0.619299f}, {0.993170f, 0.870024f, 0.626189f}, {0.992831f, 0.877168f, 0.633109f},
    {0.992440f, 0.884330f, 0.640099f}, {0.992089f, 0.891470f, 0.647116f}, {0.991688f, 0.898627f, 0.654202f}, {0.991332f, 0.905763f, 0.661309f},
    {0.990930f, 0.912915f, 0.668481f}, {0.990570f, 0.920049f, 0.675675f}, {0.990175f, 0.927196f, 0.682926f}, {0.989815f, 0.934329f, 0.690198f},
    {0.989434f, 0.941470f, 0.697519f}, {0.989077f, 0.948604f, 0.704863f}, {0.988717f, 0.955742f, 0.712242f}, {0.988367f, 0.962878f, 0.719649f},
    {0.988033f, 0.970012f, 0.727077f}, {0.987691f, 0.977154f, 0.734536f}, {0.987387f, 0.984288f, 0.742002f}, {0.987053f, 0.991438f, 0.749504f}
};

void image::remapToMagma()
{
    for (auto& pixel : mPixels)
    {
        int index = int(std::min(std::max(pixel.x, 0.0f), 1.0f) * 255.0f);
        pixel = magmaMap[index];
    }
}

} // namespace flip_detail

struct FLIPOptions {
    float ppd = 0;
    float monitorDistance = 0.7f;
    float monitorWidth = 0.7f;
    float monitorResolutionX = 3840.0f;
};

void ComputeFLIPError(const float *testRGB, const float *referenceRGB,
                      float *flipError, int xRes, int yRes, const FLIPOptions &options) {
    using namespace flip_detail;

    if (options.ppd > 0)
        gPPD = options.ppd;
    else {
        gMonitorDistance = options.monitorDistance;
        gMonitorWidth = options.monitorWidth;
        gMonitorResolutionX = options.monitorResolutionX;
        gPPD = calculatePPD(gMonitorDistance, gMonitorResolutionX, gMonitorWidth);
    }

    image testImage(xRes, yRes), referenceImage(xRes, yRes), flipImage(xRes, yRes);
    for (int y = 0; y < yRes; ++y)
        for (int x = 0; x < xRes; ++x) {
            int offset = 3 * (x + y * xRes);
            testImage.set(x, y, color3(testRGB[offset], testRGB[offset+1], testRGB[offset+2]));
            referenceImage.set(x, y, color3(referenceRGB[offset], referenceRGB[offset+1], referenceRGB[offset+2]));
        }

    flipImage.computeFLIPError(testImage, referenceImage, false /* verbose */);

    for (int y = 0; y < yRes; ++y)
        for (int x = 0; x < xRes; ++x)
            flipError[x + y * xRes] = flipImage.get(x, y).x;
}

