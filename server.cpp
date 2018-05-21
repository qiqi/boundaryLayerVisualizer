#include<cinttypes>
#include<cassert>
#include<cstdio>
#include<vector>

#include<unistd.h>
#include<sys/socket.h>
#include<netinet/in.h>

#define nGrids 160

int numProfiles()
{
    FILE * fp = fopen("data/all.bin", "rb");
    fseek(fp, 0L, SEEK_END);
    long int nBytes = ftell(fp);
    fclose(fp);

    assert(nBytes % (nGrids * sizeof(float)) == 0);
    return nBytes / nGrids / sizeof(float);
}

float * loadToGPU(int nProfiles)
{
    float *dataCPU = new float[nProfiles * nGrids];
    printf("Allocated space for %d profiles\n", nProfiles);

    FILE * fp = fopen("data/all.bin", "rb");
    for (int32_t i = 0; i < nProfiles; ++i) {
        assert(fread(dataCPU + i * nGrids, sizeof(float) * nGrids, 1, fp));
    }
    fclose(fp);
    printf("Read all profiles\n");

    float * dataGPU;
    cudaMalloc(&dataGPU, nProfiles * nGrids * sizeof(float));
    cudaMemcpy(dataGPU, dataCPU, nProfiles * nGrids * sizeof(float),
               cudaMemcpyHostToDevice);
    delete[] dataCPU;
    printf("Copied %ld bytes to GPU\n", nProfiles * nGrids * sizeof(float));

    return dataGPU;
}

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
ProfileStats computeStats(const float * u)
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
void _computeStats(ProfileStats * stats, const float * profileData, int nProf)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nProf) {
        stats[i] = computeStats(profileData + nGrids * i);
    }
}

ProfileStats * computeStats(const float * profileData, int nProf)
{
    ProfileStats * stats;
    cudaMalloc(&stats, sizeof(ProfileStats) * nProf);
    int threadsPerBlock = 128;
    int nBlocks = (int)ceil((float)nProf / threadsPerBlock);
    _computeStats<<<nBlocks, threadsPerBlock>>>(stats, profileData, nProf);
    return stats;
}

__global__
void _zeroView(float * res, int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nx * ny * 3) {
        res[i] = 0.0f;
    }
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

__global__
void _updateView(float * res, const ProfileStats * statsGPU,
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

struct ViewPort {
    float x0, x1, y0, y1;
};

bool isValidViewPort(const ViewPort & port) {
    return (port.x0 < port.x1 and port.y0 < port.y1);
}

class View {
    const int nx, ny;
    float * resGPU, * resCPU;

    const int nProf;
    const ProfileStats * statsGPU;

    int iStatX;
    int iStatY;

    public:

    void setStats(int statX, int statY)
    {
        if (statX >= 0 and statX < sizeof(ProfileStats) / sizeof(float))
            iStatX = statX;
        if (statY >= 0 and statY < sizeof(ProfileStats) / sizeof(float))
            iStatY = statY;
    }

    View(int nx, int ny, const ProfileStats * statsGPU, int nProf)
        : nx(nx), ny(ny), statsGPU(statsGPU), nProf(nProf), iStatX(0), iStatY(1)
    {
        cudaMalloc(&resGPU, sizeof(float) * 3 * nx * ny);
        resCPU = new float[3 * nx * ny];
    }

    ~View() {
        cudaFree(resGPU);
        delete[] resCPU;
    }

    int dataSizeBytes() const {
        return sizeof(float) * 3 * nx * ny;
    }

    const float * update(const ViewPort& viewPort)
    {
        int threadsPerBlock = 256;
        int nBlocks = (int)ceil(nx * ny * 3.0f / threadsPerBlock);
        _zeroView<<<nBlocks, threadsPerBlock>>>(resGPU, nx, ny);
        nBlocks = (int)ceil((float)nProf / threadsPerBlock);
        _updateView<<<nBlocks, threadsPerBlock>>>(
                resGPU, statsGPU, iStatX, iStatY, nProf,
                Discretization{nx, viewPort.x0, viewPort.x1},
                Discretization{ny, viewPort.y0, viewPort.y1});
        cudaMemcpy(resCPU, resGPU, dataSizeBytes(), cudaMemcpyDeviceToHost);
        float maxData = 0.0f;
        for (int i = 0; i < nx * ny * 3; ++ i) maxData = std::max(maxData, resCPU[i]);
        printf("Max = %f\n", maxData);
        return resCPU;
    }

};

void readall(int fd, void * buf, size_t count) {
    char * bufBytes = (char *) buf;
    while (count > 0) {
        ssize_t nRead = read(fd, bufBytes, count);
        if (nRead < 0) throw("Socket read error");
        count -= nRead;
        bufBytes += nRead;
    }
}

void writeall(int fd, const void * buf, size_t count) {
    const char * bufBytes = (const char *) buf;
    while (count > 0) {
        ssize_t nWrote = write(fd, bufBytes, count);
        if (nWrote < 0) throw("Socket write error");
        count -= nWrote;
        bufBytes += nWrote;
    }
}

class Connection
{
    private:
    int sockfd;

    public:

    Connection(int portno)
    {
        printf("Serving port %d\n", portno);
        int tmp_sockfd = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in serv_addr = {};
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_addr.s_addr = INADDR_ANY;
        serv_addr.sin_port = htons(portno);
        if (bind(tmp_sockfd, (sockaddr *) &serv_addr, sizeof(serv_addr)) < 0)
            throw("ERROR on binding");
        listen(tmp_sockfd, 5);
        printf("READY\n");
        fflush(stdout);
        sockaddr_in cli_addr;
        socklen_t clilen = sizeof(cli_addr);
        sockfd = accept(tmp_sockfd, (struct sockaddr *) &cli_addr, &clilen);
        if (sockfd < 0) throw("ERROR on accept");
    }

    ViewPort receive() const {
        ViewPort port;
        readall(sockfd, &port, sizeof(port));
        return port;
    }

    void send(const void * data, uint64_t nBytes) {
        printf("sending %ld bytes\n", nBytes);
        writeall(sockfd, data, nBytes);
    }
};

int main(int argc, char * args[])
{
    int nProf = numProfiles();
    float * profileData = loadToGPU(nProf);
    ProfileStats * stats = computeStats(profileData, nProf);

    int nx = (argc > 1) ? atoi(args[1]) : 512;
    int ny = (argc > 2) ? atoi(args[2]) : nx;
    View view(nx, ny, stats, nProf);

    int portno = (argc > 3) ? atoi(args[3]) : 18888;
    try {
        Connection connection(portno);
    
        while(true) {
            const ViewPort viewPort = connection.receive();
            fprintf(stderr, "(%f,%f)x(%f,%f)\n",
                    viewPort.x0, viewPort.x1, viewPort.y0, viewPort.y1);
            if (not isValidViewPort(viewPort)) break;
            const float * res = view.update(viewPort);
            connection.send(res, view.dataSizeBytes());
        }
    } catch(const char * e) {
        fprintf(stderr, "%s\n", e);
    }
    return 0;
}
