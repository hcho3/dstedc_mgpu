#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <matio.h>
#include "dstedc.h"

double *read_mat(const char *filename, const char *varname, size_t *dims)
{
    mat_t *mat;
    matvar_t *var;
    double *array;
    size_t nbytes;

    mat = Mat_Open(filename, MAT_ACC_RDONLY);
    if (!mat) {
        fprintf(stderr, "Error opening MAT file \"%s\"!\n", filename);
        exit(EXIT_FAILURE);
    }

    var = Mat_VarRead(mat, varname);

    if (!var) {
        fprintf(stderr, "Error reading the variable %s.\n", varname);
        Mat_Close(mat);
        exit(EXIT_FAILURE);
    }

    if (var->data_type != MAT_T_DOUBLE || var->class_type != MAT_C_DOUBLE
        || var->rank != 2) {
        fprintf(stderr, "%s is not a double-precision matrix.\n", varname);
        Mat_VarFree(var);
        Mat_Close(mat);
        exit(EXIT_FAILURE);
    }

    nbytes = var->dims[0] * var->dims[1] * sizeof(double);
    array = (double *)malloc(nbytes);
    memcpy(array, var->data, nbytes);
    memcpy(dims, var->dims, 2 * sizeof(size_t));

    Mat_VarFree(var);
    Mat_Close(mat);

    return array;
}

void write_mat(const char *filename, const char *varname,
    double *array, size_t *dims)
{
    mat_t *mat;
    matvar_t *var;
    
    mat = Mat_CreateVer(filename, NULL, MAT_FT_DEFAULT);
    if (!mat) {
        fprintf(stderr, "Error creating MAT file \"%s\"\n", filename);
        exit(EXIT_FAILURE);
    }

    var = Mat_VarCreate(varname, MAT_C_DOUBLE, MAT_T_DOUBLE,
                        2, dims, array, 0);
    if (!var) {
        fprintf(stderr, "Error creating variable %s.\n", varname);
        Mat_Close(mat);
        exit(EXIT_FAILURE);
    }

    Mat_VarWrite(mat, var, MAT_COMPRESSION_NONE);
    Mat_VarFree(var);
    Mat_Close(mat);
}
