template<int nGrids>
struct ProfileStats
{
    union {
        struct{
            float H, Hp, tau;
            float HpSubList[nGrids];
            float thetaList[nGrids];
            float tauList[nGrids];
        };
        float data[1];
    };
};

template<int nGrids>
__device__ __forceinline__
void computeIblStats(ProfileStats<nGrids>& stats, const float * u)
{
    stats.tau = u[1];
    float theta = 0.0f, deltaStar = 0.0f, thetaStar = 0.0f;
    for (int i = 1; i < nGrids; ++i) {
        float xGauss = 0.5773502691896258;
        float uGaussL = (u[i] * (1-xGauss) + u[i-1] * (1+xGauss)) / 2;
        float uGaussR = (u[i] * (1+xGauss) + u[i-1] * (1-xGauss)) / 2;
        deltaStar += 1 - (uGaussL + uGaussR) / 2;
        theta += (uGaussL - uGaussL*uGaussL + uGaussR - uGaussR*uGaussR) / 2;
        thetaStar += (uGaussL*uGaussL - uGaussL*uGaussL*uGaussL
                    + uGaussR*uGaussR - uGaussR*uGaussR*uGaussR) / 2;
    }
    stats.H = deltaStar / theta;
    float mean1minusU = deltaStar / (nGrids - 1);
    float mean1minusU2 = (deltaStar - theta) / (nGrids - 1);
    float mean1minusU3 = (deltaStar - 2 * theta + thetaStar) / (nGrids - 1);
    float mean_UminusMeanU_sq = mean1minusU2 - mean1minusU * mean1minusU;
    float mean_UminusMeanU_cu = mean1minusU3
                              - 3 * mean1minusU2 * mean1minusU;
                              + 2 * mean1minusU * mean1minusU * mean1minusU;
    stats.Hp = mean_UminusMeanU_cu / mean_UminusMeanU_sq;
}

template<int nGrids>
__device__ __forceinline__
void computeSubStats(ProfileStats<nGrids>& stats, const float * u)
{
    float sumU1 = 0.0f, sumU2 = 0.0f, sumU3 = 0.0f;
    for (int i = 1; i < nGrids; ++i) {
        float xGauss = 0.5773502691896258;
        float uGaussL = (u[i] * (1-xGauss) + u[i-1] * (1+xGauss)) / 2;
        float uGaussR = (u[i] * (1+xGauss) + u[i-1] * (1-xGauss)) / 2;
        sumU1 += (uGaussL + uGaussR) / 2;
        sumU2 += (uGaussL*uGaussL + uGaussR*uGaussR) / 2;
        sumU3 += (uGaussL*uGaussL*uGaussL + uGaussR*uGaussR*uGaussR) / 2;
        float meanU1 = sumU1 / i;
        float meanU2 = sumU2 / i;
        float meanU3 = sumU3 / i;
        float mean_UminusMeanU_sq = meanU2 - meanU1 * meanU1;
        float mean_UminusMeanU_cu = meanU3 - 3 * meanU2 * meanU1;
                                  + 2 * meanU1 * meanU1 * meanU1;
        stats.HpSubList[i] = mean_UminusMeanU_cu
                           / mean_UminusMeanU_sq
                           / sqrt(meanU2);
        stats.thetaList[i] = mean_UminusMeanU_sq / meanU2;
        stats.tauList[i] = u[1] / i * 2;
    }
}

template<int nGrids>
__device__ __forceinline__
ProfileStats<nGrids> computeStats(const float * u)
{
    ProfileStats<nGrids> stats = {};
    computeIblStats(stats, u);
    computeSubStats(stats, u);
    return stats;
}

template<int nGrids>
__global__
void _computeStats(ProfileStats<nGrids> * stats, const float * profileData,
                   int nProf)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nProf) {
        stats[i] = computeStats<nGrids>(profileData + nGrids * i);
    }
}

template<int nGrids>
ProfileStats<nGrids> * computeStats(const float * profileData, int nProf)
{
    ProfileStats<nGrids> * stats;
    cudaMalloc(&stats, sizeof(ProfileStats<nGrids>) * nProf);
    int threadsPerBlock = 128;
    int nBlocks = (int)ceil((float)nProf / threadsPerBlock);
    _computeStats<nGrids><<<nBlocks, threadsPerBlock>>>(
            stats, profileData, nProf);
    return stats;
}

