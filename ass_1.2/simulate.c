#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "simulate.h"

#define C 0.15

double *simulate(const int i_max, const int t_max, const int num_threads,
                 double *old_array, double *current_array, double *next_array)
{
    int i, t;
    double *old = old_array;
    double *current = current_array;
    double *next = next_array;
    
    // Number of threads in parallel 
    omp_set_num_threads(num_threads);
    
    for (t = 0; t < t_max; t++) {
        
        #pragma omp parallel for schedule(static) private(i)
        // Main Computation
        for (i = 1; i < i_max - 1; i++) {
            next[i] = 2.0 * current[i] - old[i] + 
                      C * (current[i-1] - 2.0 * current[i] + current[i+1]);
        }
        
        // Boundaries are 0
        next[0] = 0.0;
        next[i_max - 1] = 0.0;
        
        // Pointer rotation
        double *temp = old;
        old = current;
        current = next;
        next = temp;
    }
    
    return current;
}