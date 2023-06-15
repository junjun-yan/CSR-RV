#include <iostream>
#include <fstream>
#include <string.h>
#include <string>
#include <time.h>

#include <sys/time.h>
// #include <Windows.h>
#include <stdlib.h>

#include <algorithm>

#include <omp.h>
#include <immintrin.h>

#define FIELD_LENGTH 128
#define floatType double
#define TEST_NUM 10000
#define WARMUP_NUM 50
#define MICRO_IN_SEC 1000000.00
#define THREAD_NUM 48

using namespace std;

double microtime()
{
    int tv_sec, tv_usec;
    double time;
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);

    return (tv.tv_sec + tv.tv_usec / MICRO_IN_SEC) * 1000; // ms
}

// double microtime()
// {
//     LARGE_INTEGER nFreq;
//     LARGE_INTEGER nTime;

//     QueryPerformanceFrequency(&nFreq);
//     QueryPerformanceCounter(&nTime);

//     return (double)(nTime.QuadPart) / (double)(nFreq.QuadPart);
// }

struct Coordinate
{
    int x;
    int y;
    float val;
};

inline int coordcmp(const void *v1, const void *v2)
{
    struct Coordinate *c1 = (struct Coordinate *)v1;
    struct Coordinate *c2 = (struct Coordinate *)v2;

    if (c1->x != c2->x)
    {
        return (c1->x - c2->x);
    }
    else
    {
        return (c1->y - c2->y);
    }
}

// ****************************************************************************
// Function: readMatrix
//
// Purpose:
//   Reads a sparse matrix from a file of Matrix Market format
//   Returns the data structures for the CSR format
//
// Arguments:
//   filename: c string with the name of the file to be opened
//   val_ptr: input - pointer to uninitialized pointer
//            output - pointer to array holding the non-zero values
//                     for the  matrix
//   cols_ptr: input - pointer to uninitialized pointer
//             output - pointer to array of column indices for each
//                      element of the sparse matrix
//   rowDelimiters: input - pointer to uninitialized pointer
//                  output - pointer to array holding
//                           indices to rows of the matrix
//   n: input - pointer to uninitialized int
//      output - pointer to an int holding the number of non-zero
//               elements in the matrix
//   size: input - pointer to uninitialized int
//         output - pointer to an int holding the number of rows in
//                  the matrix
//
// Programmer: Lukasz Wesolowski
// Creation: July 2, 2010
// Returns:  nothing directly
//           allocates and returns *val_ptr, *cols_ptr, and
//           *rowDelimiters_ptr indirectly
//           returns n and size indirectly through pointers
// ****************************************************************************

void readMatrix(string filename, floatType **val_ptr, int **cols_ptr,
                int **rowDelimiters_ptr, int *n, int *numRows, int *numCols)
{
    std::string line;
    char id[FIELD_LENGTH];
    char object[FIELD_LENGTH];
    char format[FIELD_LENGTH];
    char field[FIELD_LENGTH];
    char symmetry[FIELD_LENGTH];

    std::ifstream mfs(filename);
    if (!mfs.good())
    {
        std::cerr << "Error: unable to open matrix file " << filename << std::endl;
        exit(1);
    }

    int symmetric = 0;
    int pattern = 0;
    int field_complex = 0;

    int nRows, nCols, nElements;

    struct Coordinate *coords;

    // read matrix header
    if (getline(mfs, line).eof())
    {
        std::cerr << "Error: file " << filename << " does not store a matrix" << std::endl;
        exit(1);
    }

    sscanf(line.c_str(), "%s %s %s %s %s", id, object, format, field, symmetry);

    if (strcmp(object, "matrix") != 0)
    {
        fprintf(stderr, "Error: file %s does not store a matrix\n", filename.c_str());
        exit(1);
    }

    if (strcmp(format, "coordinate") != 0)
    {
        fprintf(stderr, "Error: matrix representation is dense\n");
        exit(1);
    }

    if (strcmp(field, "pattern") == 0)
    {
        pattern = 1;
    }

    if (strcmp(field, "complex") == 0)
    {
        field_complex = 1;
    }

    if (strcmp(symmetry, "symmetric") == 0)
    {
        symmetric = 1;
    }

    while (!getline(mfs, line).eof())
    {
        if (line[0] != '%')
        {
            break;
        }
    }

    // read the matrix size and number of non-zero elements
    sscanf(line.c_str(), "%d %d %d", &nRows, &nCols, &nElements);

    int valSize = nElements * sizeof(struct Coordinate);

    if (symmetric)
    {
        valSize *= 2;
    }

    //    coords = new Coordinate[valSize];
    coords = (struct Coordinate *)malloc(valSize);

    int index = 0;
    float xx99 = 0;
    while (!getline(mfs, line).eof())
    {
        if (pattern)
        {
            sscanf(line.c_str(), "%d %d", &coords[index].x, &coords[index].y);
            coords[index].val = index % 13;
        }
        else if (field_complex)
        {
            // read the value from file
            sscanf(line.c_str(), "%d %d %f %f", &coords[index].x, &coords[index].y,
                   &coords[index].val, &xx99);
        }
        else
        {
            // read the value from file
            sscanf(line.c_str(), "%d %d %f", &coords[index].x, &coords[index].y,
                   &coords[index].val);
        }

        coords[index].x -= 1;
        coords[index].y -= 1;
        index++;

        // add the mirror element if not on main diagonal
        if (symmetric && coords[index - 1].x != coords[index - 1].y)
        {
            coords[index].x = coords[index - 1].y;
            coords[index].y = coords[index - 1].x;
            coords[index].val = coords[index - 1].val;
            index++;
        }
    }

    nElements = index;

    std::cout << "===========================================================================" << std::endl;
    std::cout << "=========*********  Informations of the sparse matrix   *********==========" << std::endl;
    std::cout << std::endl;
    std::cout << "     Number of Rows is :" << nRows << std::endl;
    std::cout << "  Number of Columns is :" << nCols << std::endl;
    std::cout << " Number of Elements is :" << nElements << std::endl;
    std::cout << std::endl;
    std::cout << "===========================================================================" << std::endl;

    std::cout << "............ Converting the Raw matrix to CSR ................." << std::endl;

    // sort the elements
    qsort(coords, nElements, sizeof(struct Coordinate), coordcmp);

    // create CSR data structures
    *n = nElements;
    *numRows = nRows;
    *numCols = nCols;

    *val_ptr = (floatType *)aligned_alloc(64, sizeof(floatType) * nElements);
    *cols_ptr = (int *)aligned_alloc(64, sizeof(int) * nElements);
    *rowDelimiters_ptr = (int *)aligned_alloc(64, sizeof(int) * (nRows + 1));

    floatType *val = *val_ptr;
    int *cols = *cols_ptr;
    int *rowDelimiters = *rowDelimiters_ptr;

    val[0] = coords[0].val;
    cols[0] = coords[0].y;
    rowDelimiters[0] = 0;

    int row = 1, i = 1;
    for (int i = 1; i < nElements; i++)
    {
        val[i] = coords[i].val;
        cols[i] = coords[i].y;
        if (coords[i].x != coords[i - 1].x)
        {
            rowDelimiters[row++] = i;
        }
    }

    rowDelimiters[row] = nElements;

    free(coords);
}

// basic
void spmv_csr(floatType *h_val, int *h_cols, int *h_rowDelimiters, int &numRows, floatType *x, floatType *y)
{
    int i, row;
    double sum;
    #pragma omp parallel private(i, row, sum)
    {
    #pragma omp for schedule(static) nowait
        for (row = 0; row < numRows; row++)
        {
            sum = 0;
            for (i = h_rowDelimiters[row]; i < h_rowDelimiters[row + 1]; i += 1)
            {
                y[row]+= h_val[i] * x[h_cols[i]];
            }
            y[row] = sum;
        }
    }
}

int main()
{
    omp_set_num_threads(THREAD_NUM);
    cout << "csr, thread num: " << omp_get_max_threads() << endl;
    string matrix = "../matrix101/juzheng1.mtx";
    string outfile = "../matrix/juzheng1_out.txt";
    std::ofstream ofs(, std::ostream::app);

    floatType *h_val;
    int *h_cols;
    int *h_rowDelimiters;

    // Number of non-zero elements in the matrix
    int nItems;
    int numRows;
    int numCols;

    floatType *x;
    floatType *y;

    readMatrix(matrix, &h_val, &h_cols, &h_rowDelimiters, &nItems, &numRows, &numCols);

    x = (floatType *)aligned_alloc(64, sizeof(floatType) * numCols);

    for (int i = 0; i < numCols; i++)
    {
        x[i] = i % 10;
    }

    y = (floatType *)aligned_alloc(64, sizeof(floatType) * numRows);
    memset(y, 0, sizeof(floatType) * numRows);

    // warm up
    for (int i = 0; i < WARMUP_NUM; i++)
    {
        spmv_csr(h_val, h_cols, h_rowDelimiters, numRows, x, y);
    }

    double kk0;
    for (int i = 0; i < TEST_NUM; i++)
    {
        kk0 = microtime();
        spmv_csr(h_val, h_cols, h_rowDelimiters, numRows, x, y);
        cout << *time = (microtime() - kk0) << endl;
    }

    cout << "The SpMV Time of csr is " << *time << " ms." << endl;

    free(h_rowDelimiters);
    free(h_cols);
    free(h_val);
    free(x);
    free(y);
}

// void test_file(string matrix, double *pre_pro_time, double *time, double *band)
// {
//     static floatType *h_val;
//     static int *h_cols;
//     static int *h_rowDelimiters;

//     // Number of non-zero elements in the matrix
//     static int nItems;
//     static int numRows;
//     static int numCols;

//     static floatType *x;
//     static floatType *y;

//     readMatrix(matrix, &h_val, &h_cols, &h_rowDelimiters, &nItems, &numRows, &numCols);

//     x = (floatType *)aligned_alloc(64, sizeof(floatType) * numCols);

//     for (int i = 0; i < numCols; i++)
//     {
//         x[i] = i % 10;
//     }

//     y = (floatType *)aligned_alloc(64, sizeof(floatType) * numRows);
//     memset(y, 0, sizeof(floatType) * numRows);

//     // warm up
//     for (int i = 0; i < WARMUP_NUM; i++)
//     {
//         spmv_csr(h_val, h_cols, h_rowDelimiters, numRows, x, y);
//     }

//     double kk0 = microtime();
//     for (int i = 0; i < TEST_NUM; i++)
//     {
//         spmv_csr(h_val, h_cols, h_rowDelimiters, numRows, x, y);
//     }

//     *time = (microtime() - kk0) / TEST_NUM;

//     cout << "The SpMV Time of csr is " << *time << " ms." << endl;

//     free(h_rowDelimiters);
//     free(h_cols);
//     free(h_val);
//     free(x);
//     free(y);
// }

// int main()
// {
//     omp_set_num_threads(THREAD_NUM);
//     cout << "csr" << endl;
//     string input_file_path = "./in/matrix101.txt";
//     string output_file_path = "./out/matrix108_csr_0326_thread48.txt";
//     string matrix_dir_path = "../matrix101/";
//     // string start_file = "TSOPF_RS_b2383.mtx";

//     std::ifstream ifs(input_file_path, std::ifstream::in);
//     std::ofstream ofs(output_file_path, std::ostream::app);

//     double pre_pro_time, time, band;
//     string file_name, file_path;

//     // while (file_name != start_file)
//     // {
//     //     ifs >> file_name;
//     // }
//     // cout << file_name << endl;
//     // file_path = matrix_dir_path + file_name;
//     // test_file(file_path, &pre_pro_time, &time, &band);
//     // ofs << file_name << " " << time << endl;

//     while (ifs >> file_name)
//     {
//         cout << file_name << endl;
//         file_path = matrix_dir_path + file_name;
//         test_file(file_path, &pre_pro_time, &time, &band);
//         ofs << file_name << " " << time << endl;
//     }

//     ofs << endl;

//     // test_file("../matrix101/A.mtx", &pre_pro_time, &time, &band);
//     // test_file("/home/forrestyan/projects/suitesparse_matrix/CO.mtx", &pre_pro_time, &time, &band);
//     // ofs << file_name << " ,csr time: " << time << "ms." << endl;
// }

// icc csr.cpp -o csr -O3 -fopenmp -std=c++11
// srun -p v5_192 -N 1 -n 1 -c 48 csr