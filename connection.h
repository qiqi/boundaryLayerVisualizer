#include<algorithm>

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
        int isTrue = 1;
        setsockopt(tmp_sockfd, SOL_SOCKET, SO_REUSEADDR, &isTrue, sizeof(int));
        sockaddr_in serv_addr = {};
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_addr.s_addr = INADDR_ANY;
        serv_addr.sin_port = htons(portno);
        if (bind(tmp_sockfd, (sockaddr *) &serv_addr, sizeof(serv_addr)) < 0)
            throw("ERROR on binding");
        listen(tmp_sockfd, 5);
        fflush(stdout);
        sockaddr_in cli_addr;
        socklen_t clilen = sizeof(cli_addr);
        sockfd = accept(tmp_sockfd, (struct sockaddr *) &cli_addr, &clilen);
        if (sockfd < 0) throw("ERROR on accept");
        close(tmp_sockfd);
    }

    ~Connection() {
        close(sockfd);
    }

    template<typename resType>
    const resType receive() const {
        uint32_t length;
        readall(sockfd, &length, sizeof(length));
        resType res;
        readall(sockfd, &res, std::min((size_t)length, sizeof(res)));
        return res;
    }

    void send(const void * data, uint64_t nBytes) {
        printf("sending %ld bytes\n", nBytes);
        writeall(sockfd, data, nBytes);
    }
};

