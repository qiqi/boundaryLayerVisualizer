struct ProfileStats
{
    union {
        struct{
            float H, tau;
        };
        float data[1];
    };
};

__device__
ProfileStats computeStats(const float * u, int nGrids)
{
    ProfileStats stats = {};
    stats.tau = u[1];
    float theta = 0.0f, deltaStar = 0.5f;
    for (int i = 1; i < nGrids; ++i) {
        deltaStar += (1 - u[i]);
        theta += u[i] * (1 - u[i]);
    }
    stats.H = deltaStar / theta;
    return stats;
}

__global__
void _computeStats(ProfileStats * stats, const float * profileData,
                   int nGrids, int nProf)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nProf) {
        stats[i] = computeStats(profileData + nGrids * i, nGrids);
    }
}

ProfileStats * computeStats(const float * profileData, int nGrids, int nProf)
{
    ProfileStats * stats;
    cudaMalloc(&stats, sizeof(ProfileStats) * nProf);
    int threadsPerBlock = 128;
    int nBlocks = (int)ceil((float)nProf / threadsPerBlock);
    _computeStats<<<nBlocks, threadsPerBlock>>>(
            stats, profileData, nGrids, nProf);
    return stats;
}

