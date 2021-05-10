/**
 * Gauss-Seidel implementation using pthreads.
 *
 *
 * Course: Advanced Computer Architecture, Uppsala University
 * Course Part: Lab assignment 3
 *
 * Original author: Fr�d�ric Haziza <daz@it.uu.se>
 * Heavily modified by: Andreas Sandberg <andreas.sandberg@it.uu.se>
 *
 */

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdatomic.h>
#include <stdbool.h>
#include "gs_interface.h"

/**
 * Tell the startup code that we want run in parallel mode.
 */
const int gsi_is_parallel = 1;

/**
 * Thread data structure passed to the thread entry function.
 */
typedef struct {
	int thread_id;
	pthread_t thread;

	/* TASK: Do you need any thread local state for synchronization? */
	atomic_flag *proceed;
	double *local_error;
} thread_info_t;

/** Define to enable debug mode */
#define DEBUG 0		/* 1 */

/** Debug output macro. Only active when DEBUG is non-0 */
#define dprintf(...)                            \
        if (DEBUG)                              \
                fprintf(stderr, __VA_ARGS__)

/** Vector with information about all active threads */
thread_info_t *threads = NULL;

/** The global error for the last iteration */
static double global_error = 0.0;

/** Synchronization barrier */
pthread_barrier_t barrier;

void gsi_init()
{
	gs_verbose_printf("\t****  Initializing the  environment ****\n");

	threads = (thread_info_t *) malloc(gs_nthreads * sizeof(thread_info_t));
	if (!threads) {
		fprintf(stderr, "Failed to allocate memory for thread information.\n");
		exit(EXIT_FAILURE);
	}

	/* Initialize global_error to something larger than the
	 * tolerance to get the algorithm started */
	global_error = gs_tolerance + 1;

	/* TASK: Initialize global variables here */
	threads[0].proceed = NULL;
	for (int i = 1; i < gs_nthreads; i++) {
		threads[i].proceed = malloc(gs_size * sizeof(atomic_flag));
		if (threads[i].proceed == NULL)
			perror("malloc");
	}

	pthread_barrier_init(&barrier, NULL, gs_nthreads);

	for (int j = 1; j < gs_nthreads; j++)
		for (int i = 0; i < gs_size; i++)
			atomic_flag_test_and_set(&threads[j].proceed[i]);
}

void gsi_finish()
{
	gs_verbose_printf("\t****  Cleaning environment ****\n");

	/* TASK: Be nice and cleanup any stuff you initialized in
	 * gsi_init()
	 */
	pthread_barrier_destroy(&barrier);

	if (threads) {
		for (int i = 1; i < gs_nthreads; i++)
			free(threads[i].proceed);
		free(threads);
	}
}

static double thread_sweep(int tid, int iter, int lbound, int rbound)
{
	double error = 0.0;

	for (int row = 1; row < gs_size - 1; row++) {
		dprintf("%d: checking wait condition\n""\titeration: %i, row: %i\n", tid, iter, row);
		
		/* TASK: Wait for data to be available from the thread to the left */
		if (tid != 0)
			while (atomic_flag_test_and_set(&threads[tid].proceed[row]));

		dprintf("%d: Starting on row: %d\n", tid, row);

		/* Update this thread's part of the matrix */
		for (int col = lbound; col < rbound; col++) {
			double new_value =
			    0.25 * (gs_matrix[GS_INDEX(row + 1, col)] +
				    gs_matrix[GS_INDEX(row - 1, col)] +
				    gs_matrix[GS_INDEX(row, col + 1)] +
				    gs_matrix[GS_INDEX(row, col - 1)]);
			
			error += fabs(gs_matrix[GS_INDEX(row, col)] - new_value);

			gs_matrix[GS_INDEX(row, col)] = new_value;
		}

		/* TASK: Tell the thread to the right that this thread is done */
		if (tid != gs_nthreads - 1)
			atomic_flag_clear(&threads[tid + 1].proceed[row]);

		dprintf("%d: row %d done\n", tid, row);
	}

	return error;
}

/**
 * Computing routine for each thread
 */
static void *thread_compute(void *_self)
{
	thread_info_t *self = (thread_info_t *) _self;
	const int tid = self->thread_id;
	int lbound = 0, rbound = 0;
	double error_ret;

	self->local_error = &error_ret;

	/* TASK: Compute bounds for this thread */
	lbound = (tid ? tid * (gs_size / gs_nthreads) : 1);
	rbound = (tid ? (lbound + (gs_size / gs_nthreads)) : (gs_size / gs_nthreads));
	if (tid == gs_nthreads - 1)
		rbound -= 1;

	gs_verbose_printf("%i: lbound: %i, rbound: %i\n", tid, lbound, rbound);

	for (int iter = 0; iter < gs_iterations && global_error > gs_tolerance; iter++) {
		dprintf("%i: Starting iteration %i\n", tid, iter);

		error_ret = thread_sweep(tid, iter, lbound, rbound);

		/* TASK: Update global error */
		/* Note: The reduction should only be done by one thread after all threads have updated their local errors */
		/* Hint: Which thread is guaranteed to complete its sweep last? */
		if (tid == gs_nthreads - 1) {
			global_error = 0.0;
			for (int i = 0; i < gs_nthreads; i++)
				global_error += *(threads[i].local_error);
		}

		dprintf("%d: iteration %d done\n", tid, iter);

		/* TASK: Iteration barrier */
		pthread_barrier_wait(&barrier);
	}

	gs_verbose_printf("\t****  Thread %d done after %d iterations ****\n",
			  tid, gs_iterations);

	return NULL;
}

/**
 * Parallel implementation of the GS algorithm. Called from
 * gs_common.c to start the solver.
 */
void gsi_calculate()
{
	int err;

	for (int t = 0; t < gs_nthreads; t++) {
		gs_verbose_printf("\tSpawning thread %d\n", t);

		threads[t].thread_id = t;
		err = pthread_create(&threads[t].thread, NULL, thread_compute, &threads[t]);
		if (err) {
			fprintf(stderr, "Error: pthread_create() failed: %d, ""thread %d\n", err, t);
			exit(EXIT_FAILURE);
		}
	}

	/* Calling pthread_join on a thread will block until the
	 * thread terminates. Since we are joining all threads, we
	 * wait until all threads have exited. */
	for (int t = 0; t < gs_nthreads; t++) {
		err = pthread_join(threads[t].thread, NULL);
		if (err) {
			fprintf(stderr, "Error: pthread_join() failed: %d, ""thread %d\n", err, t);
			exit(EXIT_FAILURE);
		}
	}

	if (global_error <= gs_tolerance)
		printf("Solution converged!\n");
	else {
		printf("Solution did NOT converge.\n");
		printf("Note: Not converging is normal if you are using the "
		       "default settings.\n");
	}
}

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 8
 * indent-tabs-mode: nil
 * c-file-style: "linux"
 * End:
 */
