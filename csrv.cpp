#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <time.h>

#define MICRO_IN_SEC 1000000.00
#include <omp.h>

// #include <sys/time.h>
// #include <unistd.h>
// #include <sys/mman.h>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <list>
#include <vector>
#include <set>
#define WRITE

#include <immintrin.h>

#define FIELD_LENGTH 128
#define floatType double

struct Coordinate
{
    int x;
    int y;
    float val;
};

// struct csrv
// {
//     int block_size;
//     csrv_block *block;
// };

// struct csrv_block
// {
//     int nnz_num;
//     int row_size, col_size;
//     double value;
//     int *row_ptr;
//     int *col_idx;
// };

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

void readMatrix(char *filename, floatType **val_ptr, int **cols_ptr,
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
        fprintf(stderr, "Error: file %s does not store a matrix\n", filename);
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

    int nElements_padding = (nElements % 16 == 0) ? nElements : (nElements + 16) / 16 * 16;

    int valSize = nElements_padding * sizeof(struct Coordinate);

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
                   &coords[index].val, xx99);
        }
        else
        {
            // read the value from file
            sscanf(line.c_str(), "%d %d %f", &coords[index].x, &coords[index].y,
                   &coords[index].val);
        }

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

    nElements_padding = (nElements % 16 == 0) ? nElements : (nElements + 16) / 16 * 16;

    std::cout << "===========================================================================" << std::endl;
    std::cout << "=========*********  Informations of the sparse matrix   *********==========" << std::endl;
    std::cout << std::endl;
    std::cout << "     Number of Rows is :" << nRows << std::endl;
    std::cout << "  Number of Columns is :" << nCols << std::endl;
    std::cout << " Number of Elements is :" << nElements << std::endl;
    std::cout << "       After Alignment :" << nElements_padding << std::endl;
    std::cout << std::endl;
    std::cout << "===========================================================================" << std::endl;

    std::cout << "............ Converting the Raw matrix to CSR ................." << std::endl;

    for (int qq = index; qq < nElements_padding; qq++)
    {
        coords[qq].x = coords[index - 1].x;
        coords[qq].y = coords[index - 1].y;
        coords[qq].val = 0;
    }

    //sort the elements
    qsort(coords, nElements_padding, sizeof(struct Coordinate), coordcmp);

    // create CSR data structures
    *n = nElements_padding;
    *numRows = nRows;
    *numCols = nCols;

    *val_ptr = (floatType *)_mm_malloc(sizeof(floatType) * nElements_padding, 64);
    *cols_ptr = (int *)_mm_malloc(sizeof(int) * nElements_padding, 64);
    *rowDelimiters_ptr = (int *)_mm_malloc(sizeof(int) * (nRows + 2), 64);

    floatType *val = *val_ptr;
    int *cols = *cols_ptr;
    int *rowDelimiters = *rowDelimiters_ptr;

    rowDelimiters[0] = 0;

    int r = 0;

    int i = 0;
    for (i = 0; i < nElements_padding; i++)
    {
        while (coords[i].x != r)
        {
            rowDelimiters[++r] = i;
        }
        val[i] = coords[i].val;
        cols[i] = coords[i].y;
    }

    for (int k = r + 1; k <= (nRows + 1); k++)
    {
        rowDelimiters[k] = i - 1;
    }

    r = 0;

    free(coords);
}

// ****************************************************************************
// Function: csr2csrv
//
// Purpose:
//   Convect a csr format to csr-v format proprecessing
//   Returns the data structures for the csr-v format
//
// Arguments:
//   h_val: 
//   h_cols: 
//   h_rowDelimiters: 
//   nItems: 
//   numRows:
//   umCols:
// Programmer: Forrest Yan
// Creation: Oct 27, 2021
// Returns:  nothing directly
// ****************************************************************************

void csr2csrv(floatType &h_val, int &h_cols, int &h_rowDelimiters, int &nItems, int &numRows, int &numCols)
{
     
}

void spmv()
{
}

void csrv2csr()
{
}

int main(int argc, char** argv)
{
    static floatType *h_val;
    static int *h_cols;
    static int *h_rowDelimiters;
    // Number of non-zero elements in the matrix
    static int nItems;
    static int numRows;
    static int numCols;

    readMatrix(argv[1], &h_val, &h_cols, &h_rowDelimiters, &nItems, &numRows, &numCols);

    system("pause");
    return 0;
}