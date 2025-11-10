#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "simulate.h"

#define C 0.15

static pthread_barrier_t barrier;
static int i_max_global;
static int t_max_global;
static double *old_global;
static double *current_global;
static double *next_global;

typedef struct {
    int thread_id;
    int start_idx;
    int end_idx;
} t_data;

void *worker_thread(void *arg)
{
    t_data *data = (t_data *)arg;
    int start = data->start_idx;
    int end = data->end_idx;
    int i, t;
    
    // Use local pointers
    double *old = old_global;
    double *current = current_global;
    double *next = next_global;
    
    for (t = 0; t < t_max_global; t++) {
        // Skip boundaries (they stay at 0)
        int compute_start = (start == 0) ? 1 : start;
        int compute_end = (end == i_max_global) ? i_max_global - 1 : end;
        
        for (i = compute_start; i < compute_end; i++) {
            next[i] = 2.0 * current[i] - old[i] + 
                      C * (current[i-1] - 2.0 * current[i] + current[i+1]);
        }
        
        // Handle boundaries if this thread is responsible
        if (start == 0) {
            next[0] = 0.0;
        }
        if (end == i_max_global) {
            next[i_max_global - 1] = 0.0;
        }
        
        // Wait for all threads to finish computing
        pthread_barrier_wait(&barrier);
        
        double *temp = old;
        old = current;
        current = next;
        next = temp;
        
        // Thread 0 updates global pointers
        if (data->thread_id == 0) {
            old_global = old;
            current_global = current;
            next_global = next;
        }
        
        // Globals must be updated before next iteration
        pthread_barrier_wait(&barrier);
    }
    
    return NULL;
}

double *simulate(const int i_max, const int t_max, const int num_threads,
        double *old_array, double *current_array, double *next_array)
{
    pthread_t *threads;
    t_data *thread_data;
    int i;
    
    i_max_global = i_max;
    t_max_global = t_max;
    old_global = old_array;
    current_global = current_array;
    next_global = next_array;
    
    pthread_barrier_init(&barrier, NULL, num_threads);
    
    threads = malloc(num_threads * sizeof(pthread_t));
    thread_data = malloc(num_threads * sizeof(t_data));
    
    // Divide work with cache-line awareness (try to align on 64-byte boundaries)
    int base_chunk = i_max / num_threads;
    int remainder = i_max % num_threads;
    int current_start = 0;
    
    for (i = 0; i < num_threads; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].start_idx = current_start;
        
        int chunk_size = base_chunk + (i < remainder ? 1 : 0);
        thread_data[i].end_idx = current_start + chunk_size;
        
        current_start = thread_data[i].end_idx;
    }
    
    // Create and run threads
    for (i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, worker_thread, &thread_data[i]);
    }
    
    for (i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    pthread_barrier_destroy(&barrier);
    free(threads);
    free(thread_data);
    
    return current_global;
}