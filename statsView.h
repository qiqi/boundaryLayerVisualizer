#pragma once

#include"checkError.h"

int nBlocks(int nThreads, int threadsPerBlock) {
    return static_cast<int>(
            ceil(static_cast<float>(nThreads) / threadsPerBlock));
}

__global__
void _zeroView(float * res, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) res[i] = 0.0f;
}

struct Discretization {
    int nx;
    float x0, x1;
};

__device__
int __map(Discretization& disc, float x) {
    float frac = (x - disc.x0) / (disc.x1 - disc.x0);
    int ix = static_cast<int>(round(frac * disc.nx));
    return (ix >= disc.nx) ? -1 : ix;
}

template<int nGrids>
__global__
void _updateView(float * res, const ProfileStats<nGrids> * statsGPU,
                 int iStatX, int iStatY, int nProf,
                 Discretization discX, Discretization discY)
{
    int iProf = blockDim.x * blockIdx.x + threadIdx.x;
    if (iProf < nProf) {
        const float * stats = statsGPU[iProf].data;
        int iX = __map(discX, stats[iStatX]);
        int iY = __map(discY, stats[iStatY]);
        if (iX >= 0 and iY >= 0) {
            float * resPixel = res + (iY * discX.nx + iX) * 3;
            atomicAdd(&resPixel[0], 1.0f);
        }
    }
}

struct Pixel {
    uint32_t ix, iy;
};

struct ViewPort {
    float x0, x1, y0, y1;
};

bool isValidViewPort(const ViewPort & port) {
    return (port.x0 < port.x1 and port.y0 < port.y1);
}

template<int nGrids>
__global__
void _isInView(int * res, const ProfileStats<nGrids> * statsGPU,
               int iStatX, int iStatY, int nProf, ViewPort viewPort)
{
    int iProf = blockDim.x * blockIdx.x + threadIdx.x;
    if (iProf < nProf) {
        float statX = statsGPU[iProf].data[iStatX];
        float statY = statsGPU[iProf].data[iStatY];
        res[iProf] = (statX >= viewPort.x0 and statX <= viewPort.x1 and 
                      statY >= viewPort.y0 and statY <= viewPort.y1);
    }
}

template<int nGrids>
class StatsView {
    const int nx, ny;
    float * resGPU, * resCPU;
    int * resIsInPixel;

    const int nProf;
    const ProfileStats<nGrids> * statsGPU;

    int iStatX;
    int iStatY;

    ViewPort currentView;

    public:

    void setStats(int statX, int statY)
    {
        if (statX >= 0 and
                statX < sizeof(ProfileStats<nGrids>) / sizeof(float))
            iStatX = statX;
        if (statY >= 0 and
                statY < sizeof(ProfileStats<nGrids>) / sizeof(float))
            iStatY = statY;
    }

    StatsView(int nx, int ny, const ProfileStats<nGrids> * statsGPU,
              int nProf)
        : nx(nx), ny(ny), statsGPU(statsGPU),
          nProf(nProf), iStatX(3+40), iStatY(3+nGrids+40)
    {
        cudaMalloc(&resGPU, sizeof(float) * 3 * nx * ny);
        resCPU = new float[3 * nx * ny];
        cudaMalloc(&resIsInPixel, sizeof(int) * nProf);
    }

    ~StatsView() {
        cudaFree(resGPU);
        delete[] resCPU;
        cudaFree(resIsInPixel);
    }

    int dataSizeBytes() const {
        return sizeof(float) * 3 * nx * ny;
    }

    const int * isInView(const ViewPort& viewPort)
    {
        int threadsPerBlock = 256;
        _isInView<nGrids>
            <<<nBlocks(nProf, threadsPerBlock), threadsPerBlock>>>(
                resIsInPixel, statsGPU, iStatX, iStatY, nProf, viewPort);
        cudaCheckError();
        return resIsInPixel;
    }

    const float * update(const ViewPort& viewPort)
    {
        currentView = viewPort;
        int threadsPerBlock = 128;
        _zeroView<<<nBlocks(nx*ny*3, threadsPerBlock), threadsPerBlock>>>(
                resGPU, nx*ny*3);
        cudaCheckError();
        _updateView<nGrids>
            <<<nBlocks(nProf, threadsPerBlock), threadsPerBlock>>>(
                resGPU, statsGPU, iStatX, iStatY, nProf,
                Discretization{nx, viewPort.x0, viewPort.x1},
                Discretization{ny, viewPort.y0, viewPort.y1});
        cudaCheckError();
        cudaMemcpy(resCPU, resGPU, dataSizeBytes(), cudaMemcpyDeviceToHost);
        float maxData = 0.0f;
        for (int i = 0; i < nx * ny * 3; ++ i) {
            maxData = std::max(maxData, resCPU[i]);
        }
        printf("Stats Max = %f\n", maxData);
        return resCPU;
    }
};
