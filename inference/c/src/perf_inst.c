#include "perf_inst.h"

/** Performance instrumentation for function statistics gathering */

int perf_readers [NUM_PERF_READERS] = { 0 };
perf_stat_t func_stats [MAX_FUNC_N] = { 0 };

void perf_init_readers() {
    assert(NUM_PERF_READERS == 6); // unrolled code to initialise 6 readers gathering stats for the perf_stat_t struct

    struct perf_event_attr attr = {0};
    attr.type = PERF_TYPE_HARDWARE;
    attr.size = sizeof(struct perf_event_attr);
    attr.disabled = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;

    /* hw_cpu_cycles */
    attr.config = PERF_COUNT_HW_CPU_CYCLES;
    perf_readers[0] = syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0);
    assert(perf_readers[0] >= 0);
    ioctl(perf_readers[0], PERF_EVENT_IOC_RESET, 0);
    ioctl(perf_readers[0], PERF_EVENT_IOC_ENABLE, 0);

    /* hw_retired_inst */
    attr.config = PERF_COUNT_HW_INSTRUCTIONS;
    perf_readers[1] = syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0);
    assert(perf_readers[1] >= 0);
    ioctl(perf_readers[1], PERF_EVENT_IOC_RESET, 0);
    ioctl(perf_readers[1], PERF_EVENT_IOC_ENABLE, 0);

    /* hw_branch_inst */
    attr.config = PERF_COUNT_HW_BRANCH_INSTRUCTIONS;
    perf_readers[2] = syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0);
    assert(perf_readers[2] >= 0);
    ioctl(perf_readers[2], PERF_EVENT_IOC_RESET, 0);
    ioctl(perf_readers[2], PERF_EVENT_IOC_ENABLE, 0);

    /* hw_branch_misses */
    attr.config = PERF_COUNT_HW_BRANCH_MISSES;
    perf_readers[3] = syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0);
    assert(perf_readers[3] >= 0);
    ioctl(perf_readers[3], PERF_EVENT_IOC_RESET, 0);
    ioctl(perf_readers[3], PERF_EVENT_IOC_ENABLE, 0);

    /* hw_cache_ref */
    attr.config = PERF_COUNT_HW_CACHE_REFERENCES;
    perf_readers[4] = syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0);
    assert(perf_readers[4] >= 0);
    ioctl(perf_readers[4], PERF_EVENT_IOC_RESET, 0);
    ioctl(perf_readers[4], PERF_EVENT_IOC_ENABLE, 0);

    /* hw_cache_misses */
    attr.config = PERF_COUNT_HW_CACHE_MISSES;
    perf_readers[5] = syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0);
    assert(perf_readers[5] >= 0);
    ioctl(perf_readers[5], PERF_EVENT_IOC_RESET, 0);
    ioctl(perf_readers[5], PERF_EVENT_IOC_ENABLE, 0);

}

void perf_close_readers() {

    for(size_t i = 0; i < NUM_PERF_READERS; ++i) {
	ioctl(perf_readers[i], PERF_EVENT_IOC_DISABLE, 0);
	close(perf_readers[i]);
    }

}

void perf_read_start(void * fn) {

    size_t fn_idx = perf_get_fn_idx(fn);
    perf_stat_t * stats = func_stats + fn_idx;

    struct read_format read_data;

    assert(NUM_PERF_READERS == 6); // unrolled code to gather data from 6 readers

    read(perf_readers[0], &read_data, sizeof(struct read_format));
    stats->d_hw_cpu_cycles = read_data.value;

    read(perf_readers[1], &read_data, sizeof(struct read_format));
    stats->d_hw_retired_inst = read_data.value;

    read(perf_readers[2], &read_data, sizeof(struct read_format));
    stats->d_hw_branch_inst = read_data.value;

    read(perf_readers[3], &read_data, sizeof(struct read_format));
    stats->d_hw_branch_misses = read_data.value;

    read(perf_readers[4], &read_data, sizeof(struct read_format));
    stats->d_hw_cache_ref = read_data.value;

    read(perf_readers[5], &read_data, sizeof(struct read_format));
    stats->d_hw_cache_misses = read_data.value;

}

void perf_read_end(void * fn) {

    size_t fn_idx = perf_get_fn_idx(fn);
    perf_stat_t * stats = func_stats + fn_idx;

    ++(stats->invocations);

    struct read_format read_data;

    assert(NUM_PERF_READERS == 6); // unrolled code to finish gahtering data from 6 readers

    read(perf_readers[0], &read_data, sizeof(struct read_format));
    stats->hw_cpu_cycles += read_data.value - stats->d_hw_cpu_cycles;

    read(perf_readers[1], &read_data, sizeof(struct read_format));
    stats->hw_retired_inst += read_data.value - stats->d_hw_retired_inst;

    read(perf_readers[2], &read_data, sizeof(struct read_format));
    stats->hw_branch_inst += read_data.value - stats->d_hw_branch_inst;

    read(perf_readers[3], &read_data, sizeof(struct read_format));
    stats->hw_branch_misses += read_data.value - stats->d_hw_branch_misses;

    read(perf_readers[4], &read_data, sizeof(struct read_format));
    stats->hw_cache_ref += read_data.value - stats->d_hw_cache_ref;

    read(perf_readers[5], &read_data, sizeof(struct read_format));
    stats->hw_cache_misses += read_data.value - stats->d_hw_cache_misses;

}

size_t perf_get_fn_idx(void * fn) {
    for(size_t idx = 0; idx < MAX_FUNC_N; ++idx) {
	if(func_stats[idx].fn == fn) {
	    return idx;
	}
	if(func_stats[idx].fn == NULL) {
	    func_stats[idx].fn = fn;
	    return idx;
	}
    }
    assert(0); // run out of functions!
}

#ifdef INSTRUMENT_MATMUL

void __cyg_profile_func_enter (void * this_fn, void * call_site) {
    perf_read_start(this_fn);
}

void __cyg_profile_func_exit  (void * this_fn, void * call_site) {
    perf_read_end(this_fn);
}

#endif /* INSTRUMENT_MATMUL */

const char * perf_get_fn_name(void * fn) {
    Dl_info info;
    dladdr(fn, &info);
    return info.dli_sname; // symbol name associated with function pointer
}


void _print_stats(perf_stat_t * stats, uint8_t average) {
    size_t divisor = (average ? stats->invocations : 1);
    printf(" HW CPU CYCLES: %" PRIu64 "\n HW RETIRED INSTRUCTIONS: %" PRIu64 "\n HW BRANCH INSTRUCTIONS: %" PRIu64 "\n"
	   " HW BRANCH MISSES: %" PRIu64 "\n HW CACHE REFERENCES: %" PRIu64 "\n HW CACHE MISSES: %" PRIu64 "\n",
	stats->hw_cpu_cycles / divisor, stats->hw_retired_inst / divisor, stats->hw_branch_inst / divisor,
	stats->hw_branch_misses / divisor, stats->hw_cache_ref / divisor, stats->hw_cache_misses / divisor);
}

void perf_print_summary() {

    puts("--- function statistics summary ---");

    size_t idx = 0;
    void * function = func_stats[0].fn;
    while(function != NULL && idx < MAX_FUNC_N) {
	printf("\n-> function: %s (%p)\n", (perf_get_fn_name(function) == NULL ? "unknown" : perf_get_fn_name(function)), function);
	printf("-> invocations: %lu\n", func_stats[idx].invocations);
	puts("-> stats (total):");
	_print_stats(func_stats + idx, 0);
	puts("\n-> stats (average):");
	_print_stats(func_stats + idx, 1);
	function = func_stats[++idx].fn;
    }

    puts("--- function statistics summary done ---");
}

