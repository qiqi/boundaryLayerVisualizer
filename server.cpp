#include<cinttypes>
#include<cassert>
#include<cstdio>
#include<vector>

#include<unistd.h>
#include<sys/socket.h>
#include<netinet/in.h>

#include"profileStats.h"
#include"statsView.h"
#include"profileView.h"
#include"connection.h"

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

struct Message {
    enum class Type : uint32_t {Exit, UpdateView, SelectProfiles} type;
    union {
        ViewPort viewPort;
    };
};

int main(int argc, char * args[])
{
    int nProf = numProfiles();
    float * profileData = loadToGPU(nProf);
    ProfileStats * stats = computeStats(profileData, nGrids, nProf);

    int nx = (argc > 1) ? atoi(args[1]) : 512;
    int ny = (argc > 2) ? atoi(args[2]) : nx;
    int nU = (argc > 3) ? atoi(args[3]) : 256;
    StatsView view(nx, ny, stats, nProf);
    ProfileView pview(nU, profileData, nGrids, nProf);

    int portno = (argc > 4) ? atoi(args[4]) : 18888;
    try {
        Connection connection(portno);
    
        while(true) {
            const Message msg = connection.receive<Message>();
            if (msg.type == Message::Type::Exit) {
                break;
            }
            else if (msg.type == Message::Type::UpdateView) {
                const float * res = view.update(msg.viewPort);
                connection.send(res, view.dataSizeBytes());
            }
            else if (msg.type == Message::Type::SelectProfiles) {
                const int * isInView = view.isInView(msg.viewPort);
                const float * res = pview.update(isInView);
                connection.send(res, pview.dataSizeBytes());
            }
        }
    } catch(const char * e) {
        fprintf(stderr, "%s\n", e);
    }
    return 0;
}
