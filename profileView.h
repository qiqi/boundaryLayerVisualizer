#include<algorithm>
#include"checkError.h"
#include"statsView.h"

__global__
void _updateProfileView(float * res, const int * isActive,
                        const float * profiles, int nU,
                        int nGrid, int nProf)
{
    Discretization disc = {nU, -0.5, 1.5};
    int iProf = blockDim.x * blockIdx.x + threadIdx.x;
    if (iProf < nProf and isActive[iProf]) {
        const float * profile = profiles + iProf * nGrid;
        for (int iGrid = 0; iGrid < nGrid; ++iGrid) {
            int iU = __map(disc, profile[iGrid]);
            atomicAdd(res + iU * nGrid + iGrid, 1.0f);
        }
    }
}

class ProfileView {
    const int nU;
    float * resGPU, * resCPU;

    const int nGrid, nProf;
    const float * profiles;

    public:

    ProfileView(int nU, const float * profiles, int nGrid, int nProf)
        : nU(nU), profiles(profiles), nGrid(nGrid), nProf(nProf)
    {
        printf("nU, nGrid = %d %d\n", nU, nGrid);
        cudaMalloc(&resGPU, sizeof(float) * nU * nGrid);
        resCPU = new float[nU * nGrid];
    }

    ~ProfileView() {
        cudaFree(resGPU);
        delete[] resCPU;
    }

    int dataSizeBytes() const {
        return sizeof(float) * nU * nGrid;
    }

    const float * update(const int * isActive)
    {
        int threadsPerBlock = 256;
        _zeroView<<<nBlocks(nU * nGrid, threadsPerBlock), threadsPerBlock>>>(
                resGPU, nU * nGrid);
        cudaCheckError();
        _updateProfileView<<<nBlocks(nProf, threadsPerBlock), threadsPerBlock>>>(
                resGPU, isActive, profiles, nU, nGrid, nProf);
        cudaCheckError();
        cudaMemcpy(resCPU, resGPU, dataSizeBytes(), cudaMemcpyDeviceToHost);
        {
        float resMax = 0.0;
        for (int i = 0; i < nU * nGrid; ++i) resMax = std::max(resMax, resCPU[i]);
        printf("Profile Max = %f\n", resMax);
        }
        return resCPU;
    }

};


