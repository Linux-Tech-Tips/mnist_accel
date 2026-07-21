#ifndef PERF_INST_H
#define PERF_INST_H

/* Need _GNU_SOURCE and dynamic linking library to get symbols from function pointers */
#define _GNU_SOURCE
#include <dlfcn.h>

#include <stdio.h>
#include <stddef.h>
#include <inttypes.h>
#include <assert.h>

#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>

/** Performance instrumentation for function statistics gathering */

#define MAX_FUNC_N 32

#define NUM_PERF_READERS 6

typedef struct {

    void * fn;

    size_t invocations;

    uint64_t hw_cpu_cycles;
    uint64_t hw_retired_inst;
    uint64_t hw_branch_inst;
    uint64_t hw_branch_misses;
    uint64_t hw_cache_ref;
    uint64_t hw_cache_misses;

    uint64_t d_hw_cpu_cycles;
    uint64_t d_hw_retired_inst;
    uint64_t d_hw_branch_inst;
    uint64_t d_hw_branch_misses;
    uint64_t d_hw_cache_ref;
    uint64_t d_hw_cache_misses;

} perf_stat_t;


extern int perf_readers [NUM_PERF_READERS];
extern perf_stat_t func_stats [MAX_FUNC_N];


/** Initialises PMU readers using syscalls, to be called once before profiling */
void perf_init_readers();

/** Closes all PMU readers opened by perf_init_readers */
void perf_close_readers();

/* As per Linux Manual page perf_event_open */
struct read_format {
    uint64_t value;
};

/** Begin tracking counters for the given function */
void perf_read_start(void * fn);

/** End tracking counters for the given function, append delta and increment number of invocations */
void perf_read_end(void * fn);

/** Get index of function under given pointer in func_stats array */
size_t perf_get_fn_idx(void * fn);

#ifdef INSTRUMENT_MATMUL

/** GCC built-in instrumentation function enter hook */
void __attribute__((no_instrument_function))
__cyg_profile_func_enter (void * this_fn, void * call_site);

/** GCC built-in instrumentation function exit hook */
void __attribute__((no_instrument_function))
__cyg_profile_func_exit  (void * this_fn, void * call_site);

#endif /* INSTRUMENT_MATMUL */

/** Helper to get function name from function pointer using dlfcn.h */
const char * perf_get_fn_name(void * fn);

/** Prints summary of gathered function stats data */
void perf_print_summary();


#endif /* PERF_INST_H */
