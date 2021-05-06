/*
 ============================================================================
 Name        : kmeans.c
 Author      : Noy Shabtay
 Description : Kmeans clustering algorithm implementation for Python C-API useage.
 ============================================================================
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "kmeans.h"

const int k;
const int n;
const int d;
const int max_iter;

static PyObject* kmeans(double**, double**);
static void copy_centroids(double**, double**);
static double squared_distance(int, int, double**, double**);
static void update_centroids(double**, int*, double**);
static int compare_centroids(double**, double**);
static void sum_and_size_init(double**, int*);
static void clusters_builder(int**, double**, double**, int k, int*);
static PyObject* py_clusters_builder(int**, int*, int);
static PyObject* labeling(int**, int*, int, int);
static void free_matrix(void**, int);
static void* failure_handler(void**, int, void**, int, void*);
static double** allocate_double_matrix(double** matrix, int dim1, int dim2, void** matrix2free, int n2, void* p);

static PyObject* kmeans_runner(PyObject* self, PyObject* args) {
	/*
	reponsible for transforming python objects into C arguments and wraps the kmeans function
	returns the kmeans results to the user in python objects
	In case of failure - returns to Python a NULL to signal that an error has occurred.
	*/

	int i;
	int j;
	int index;
	double **observations;
	double **centroids;
	long *indices;
	PyObject* py_observations;
	PyObject* py_indices;
	PyObject* py_item;
	PyObject* py_dim;
	PyObject* py_answer;

	if (!PyArg_ParseTuple(args, "iiiiOO:kmeans_runner", &k, &n, &d, &max_iter, &py_indices, &py_observations)) {
		return NULL;
	}

	// matrixes memory allocations
	observations = allocate_double_matrix(observations, n, d, (void**)NULL, 0, (void*)NULL);
	if (observations == NULL) {return NULL;}
	centroids = allocate_double_matrix(centroids, k, d, (void**)observations, n, (void*)NULL);
	if (centroids == NULL) {return NULL;}

	// collecting data from the Python program incl. error handling.
 	if (!PyList_Check(py_observations)) {
		return failure_handler((void**)observations, n, (void**)centroids, k, (void*)NULL); }
	for (i = 0; i < n; i++) {
		py_item = PyList_GetItem(py_observations, i);
		if (!PyList_Check(py_item)) {
			return failure_handler((void**)observations, n, (void**)centroids, k, (void*)NULL); }
		for (j = 0; j < d; j++) {
			py_dim = PyList_GetItem(py_item, j);
			observations[i][j] = PyFloat_AsDouble(py_dim);
			if (observations[i][j]  == -1.0 && PyErr_Occurred()) {
				return failure_handler((void**)observations, n, (void**)centroids, k, (void*)NULL); }
		}
	}
	indices = (long*)malloc(k * sizeof(long));
	if (indices == NULL) {return failure_handler((void**)observations, n, (void**)centroids, k, (void*)NULL);}
	if (!PyList_Check(py_indices)) {
		return failure_handler((void**)observations, n, (void**)centroids, k, (void*)indices); }
	for (i = 0; i < k; i++) {
		py_item = PyList_GetItem(py_indices, i);
		indices[i] = PyLong_AsLong(py_item);
		if (indices[i]  == -1 && PyErr_Occurred()) {
            return failure_handler((void**)observations, n, (void**)centroids, k, (void*)indices);
         }
	}
	// setting kmeans++ chosen initialization from Python
	for (j = 0; j < k; j++) {
		index = indices[j];
		for (i = 0; i < d; i++) {
			centroids[j][i] = observations[index][i];
		}
	}
	// running kmeans implementation
	free(indices);
	py_answer = kmeans(observations, centroids);
	free_matrix((void**)centroids, k);
	free_matrix((void**)observations, n);
	return py_answer;
}

static PyObject* kmeans(double** observations, double** centroids) {
	/* 
	the main function for clustering data points using kmeans method.
	handling the algorithm's workflow
	returns a python object composed of two lists, the first one holds data points labeling, the second one holds the clusters where
	each cluster is a list of indices assigned to it 
	*/
	int i;
	int j;
	int obs;
	int iter;
	int* clusters_size;
	double min_d;
	int min_cluster;
	double **prev_centroids;
	double **clusters_sum;
	double distance;
	int** clusters;
	int* cluster_current_idx;
	PyObject* py_clusters;
	PyObject* py_Y;
	PyObject* py_answer;

	// memory allocations.
	prev_centroids = allocate_double_matrix(prev_centroids, k, d, (void**)NULL, 0, (void*)NULL);
	if (prev_centroids == NULL) {return NULL;}
	clusters_size = malloc(k*sizeof(int));
	if (clusters_size == NULL) {return failure_handler((void**)prev_centroids, k, (void**)NULL, 0, (void*)clusters_size);}
	clusters_sum = allocate_double_matrix(clusters_sum, k, d, (void**)prev_centroids, k, (void*)clusters_size);
	if (clusters_sum == NULL) {return NULL;}

	// kmeans method implementation
	for(iter=0; iter<max_iter; iter++) {
		copy_centroids(centroids ,prev_centroids); //saving the current state for future comparison
		sum_and_size_init(clusters_sum, clusters_size);
		for(obs=0; obs<n; obs++){
			min_d = 1.0 / 0.0;
			min_cluster = -1;
			for(j=0; j<k; j++){
				distance = squared_distance(j, obs, observations, centroids);
				if (distance < min_d) {
					min_d = distance;
					min_cluster = j;
				}
			}
			clusters_size[min_cluster]++;
			for (i=0; i<d; i++) {
				clusters_sum[min_cluster][i] += observations[obs][i];
			}
		}
		update_centroids(centroids, clusters_size, clusters_sum);
		if (compare_centroids(prev_centroids, centroids)) {
			break; }
	}
	free_matrix((void**)clusters_sum, k);

	// building clusters for future clusters.txt file
	clusters = (int**)malloc(k*sizeof(int*));
	if (clusters == NULL) {return failure_handler((void**)prev_centroids, k, (void**)clusters, 0, (void*)clusters_size);}
	for(j=0; j<k; j++) {
		clusters[j] = (int*)malloc(clusters_size[j]*sizeof(int));
		if (clusters[j] == NULL) {return failure_handler((void**)prev_centroids, k, (void**)clusters, j, (void*)clusters_size);}
	}
	cluster_current_idx = (int*)calloc(k, sizeof(int));
	if (cluster_current_idx == NULL) {return failure_handler((void**)prev_centroids, k, (void**)clusters, k, (void*)clusters_size);}
	clusters_builder(clusters, observations, centroids, k, cluster_current_idx);
	free(cluster_current_idx);
	
	// building python objects to return.
	py_clusters = py_clusters_builder(clusters, clusters_size, k);
	py_Y = labeling(clusters, clusters_size, k, n);
	py_answer = PyList_New(2);
	if (py_answer != NULL) {
		PyList_SetItem(py_answer, 0, py_clusters);
		PyList_SetItem(py_answer, 1, py_Y);
	}
	free(clusters_size);
	free_matrix((void**)clusters, k);
	free_matrix((void**)prev_centroids, k);
	return py_answer;
}

static void copy_centroids(double** centroids ,double** prev_centroids) {
	/*
	responsible for copying centroids by value into a new array
	*/
	int i;
	int j;
	for(j=0; j<k; j++) {
		for(i=0; i<d; i++) {
			prev_centroids[j][i] = centroids[j][i];
		}
	}
}

static double squared_distance(int j, int obs, double** observations, double** centroids) {
	/*
	calculates a squared euclidian distance between an observation and a centroid
	*/ 
	int i;
	double sum_of_squares = 0;
	double tmp;
	for(i=0; i<d; i++) {
		tmp = observations[obs][i]-centroids[j][i];
		sum_of_squares += tmp*tmp;
	}
	return sum_of_squares;
}

static void update_centroids(double** centroids, int* clusters_size, double** clusters_sum) {
	/*
	responsible for updating the values of the centroids
	*/
	int i;
	int j;
	for(j=0; j<k; j++) {
		for(i=0; i<d; i++) {
			centroids[j][i] = (clusters_sum[j][i] / clusters_size[j]);
		}
	}
}

static int compare_centroids(double** prev_centroids, double** centroids) {
	/* 
	returns a boolean result wether or not two centroids are the same
	*/
	int i;
	int j;
	for(j=0; j<k; j++) {
		for(i=0; i<d; i++) {
			if (prev_centroids[j][i] != centroids[j][i]) {
				return 0;
			}
		}
	}
	return 1;
}

static void sum_and_size_init(double** clusters_sum, int* clusters_size) {
	/*
	initiates a zero-filled array for sizes of clusters and a matrix of clusters sums
	*/
	int i;
	int j;
	for(j=0; j<k; j++) {
		clusters_size[j] = 0;
		for (i=0; i<d; i++) {
				clusters_sum[j][i] = 0;
			}
	}
}

static void clusters_builder(int** clusters, double** observations, double** centroids, int k, int* cluster_current_idx) {
	/*
	responsible for building the final matrix of clusters after reaching the final centroids
	*/
	int obs;
	int j;
	double min_d;
	int min_cluster;
	double distance;
	
	for(obs=0; obs<n; obs++){
			min_d = 1.0 / 0.0;
			min_cluster = -1;
			for(j=0; j<k; j++) {
				distance = squared_distance(j, obs, observations, centroids);
				if (distance < min_d) {
					min_d = distance;
					min_cluster = j;
				}
			}
			clusters[min_cluster][cluster_current_idx[min_cluster]] = obs;
			cluster_current_idx[min_cluster]++;
		}
}

static PyObject* py_clusters_builder(int** clusters, int* clusters_size, int k) {
	/*
	responsible for building the clusters matrix as a python list  
	*/
	int j;
	int i;
	PyObject* py_a_cluster;
	PyObject* python_int;
	PyObject* py_clusters = PyList_New(k);
	if (py_clusters == NULL) {return NULL;}
    for (j=0; j<k; j++)
    {
        py_a_cluster = PyList_New(clusters_size[j]);
		if (py_a_cluster == NULL) {break;} //NULL will raise an exception on Python.
		for (i=0; i<clusters_size[j]; i++)
    	{
        	python_int = Py_BuildValue("i", clusters[j][i]); //on failure Python will get a NULL.
        	PyList_SetItem(py_a_cluster, i, python_int);
		}
		PyList_SetItem(py_clusters, j, py_a_cluster);
    }
	return py_clusters;
}

static PyObject* labeling(int** clusters, int* clusters_size, int k, int n) {
	/*
	responsible for building the data points' labels as a python list  
	*/
	int j;
	int i;
	PyObject* python_int;
	PyObject* py_Y = PyList_New(n);
	if (py_Y == NULL) {return NULL;}
	int* Y = malloc(n*sizeof(int));
	if (Y == NULL) {return NULL;}
	
	for(j=0; j<k; j++) {
		for(i=0; i<clusters_size[j]; i++) {
			Y[clusters[j][i]] = j;
		}
	}

	for (i=0; i<n; i++) {
        python_int = Py_BuildValue("i", Y[i]); //on failure Pyhon will get a NULL.
        PyList_SetItem(py_Y, i, python_int);
    }

	free(Y);
	return py_Y;
}

static double** allocate_double_matrix(double** matrix, int dim1, int dim2, void** matrix2free, int n2, void* p) {
	int i;
	matrix = (double**)malloc(dim1*sizeof(double*));
	if (matrix == NULL) {return (double**)failure_handler((void**)matrix, 0, matrix2free, n2, p);}
	for(i=0; i<dim1; i++) {
		matrix[i] = (double*)malloc(dim2*sizeof(double));
		if (matrix[i] == NULL) {return (double**)failure_handler((void**)matrix, i, matrix2free, n2, p);}
		}
	return matrix;
}

static void free_matrix(void** matrix, int size) {
	/*
	frees memory allocated for a matrix (array of arrays)
	*/
	int i;
	for(i=0; i<size; i++){
		free(matrix[i]);
	}
	free(matrix);
}

static void* failure_handler(void** matrix1, int n1, void** matrix2, int n2, void* p) {
	/*
	frees memory allocated in case of a system failure.
	return NULL to inform that there was a probelm.
	*/
	if (matrix1 != NULL) {free_matrix(matrix1, n1);}
	if (matrix2 != NULL) {free_matrix(matrix2, n2);}
	if (p != NULL) {free(p);}
	return NULL;
}

static PyMethodDef kmeansMethods[] = {
	/* This array tells Python what methods this module has */
	{"kmeans_runner",
	  (PyCFunction)kmeans_runner,
	  METH_VARARGS,
	  PyDoc_STR("A C Kmeans implementation")},
	{NULL, NULL, 0, NULL}
};


static struct PyModuleDef moduledef = {
	/* This initiates the module using the above definitions */
	PyModuleDef_HEAD_INIT,
	"mykmeanssp", 
	NULL, 
	-1,  
	kmeansMethods 
};


PyMODINIT_FUNC
PyInit_mykmeanssp(void)
{
	return PyModule_Create(&moduledef);
}
