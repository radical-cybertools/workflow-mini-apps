#include <omp.h>
#include <mpi.h>
#include <hdf5.h>
#include <iostream>
#include <vector>
#include <string>
#include <unistd.h>
#include <string>
#include <nccl.h>



/////////////////
//io
/////////////////

void Init(int argc, char** argv)
{
    MPI_Init_thread(&argc, &argv);
}

void writeSingleRank(size_t num_bytes, const std::string& data_root_dir) 
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) 
    {
        std::string filename = data_root_dir + "/data.h5";
        size_t num_elem = num_bytes / sizeof(float);
        std::vector<float> data(num_elem);

        hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        hsize_t dims[1] = { num_elem };
        hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
        hid_t dataset_id = H5Dcreate2(file_id, "data", H5T_IEEE_F32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        H5Fclose(file_id);
    }
}

void writeNonMPI(size_t num_bytes, const std::string& data_root_dir) 
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char filename[256];
    std::sprintf(filename, "%s/data_%d.h5", data_root_dir.c_str(), rank);

    size_t num_elem = num_bytes / sizeof(float);
    std::vector<float> data(num_elem);

    hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hsize_t dims[1] = { num_elem };
    hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
    hid_t dataset_id = H5Dcreate2(file_id, "data", H5T_IEEE_F32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data())
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);
}

void MPIallReduce(const std::string& device, size_t data_size) 
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (device == "cpu") 
    {
        std::vector<float> sendbuf(data_size);
        std::vector<float> recvbuf(data_size);
        MPI_Allreduce(sendbuf.data(), recvbuf.data(), data_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    } 
    else if (device == "gpu") 
    {
        assert(0 && "Error! Currently MPI comm does not support GPU!\n");
//        ncclUniqueId id;
//        ncclComm_t comm_nccl;
//
//        if (rank == 0) ncclGetUniqueId(&id);
//        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
//
//        ncclCommInitRank(&comm_nccl, size, id, rank);
//
//        float* d_sendbuf;
//        float* d_recvbuf;
//        cudaMalloc(&d_sendbuf, data_size * sizeof(float));
//        cudaMalloc(&d_recvbuf, data_size * sizeof(float));
//        ncclAllReduce(d_sendbuf, d_recvbuf, data_size, ncclFloat, ncclSum, comm_nccl, 0);
//        cudaDeviceSynchronize();
//
//        cudaFree(d_sendbuf);
//        cudaFree(d_recvbuf);
//        ncclCommDestroy(comm_nccl);
    } 
}

void dataCopyH2D(size_t data_size) 
{
    std::vector<float> data_h(data_size);
    float* data_d;

#pragma omp target data map(alloc: data_d[0:data_size])
    {
        #pragma omp target
        {
            for (size_t i = 0; i < data_size; ++i) {
                data_d[i] = data_h[i];
            }
        }
    }
}

void axpy(const std::string& device, size_t size) 
{
    std::vector<float> x(size, 1.0f);
    std::vector<float> y(size, 2.0f);

#pragma omp target data map(to: x[0:size]) map(tofrom: y[0:size])
    {
        #pragma omp target
        {
            for (size_t i = 0; i < size; ++i) {
                y[i] += 1.01f * x[i];
            }
        }
    }
}
