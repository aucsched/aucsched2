/*****************************************************************************
 *  aucsched.c - linear programming based scheduler plugin.
 *
 *  Developed by Seren Soner and Can Ozturan 
 *  from Bogazici University, Istanbul, Turkey
 *  as a part of PRACE WP9.2
 *  Contact seren.soner@gmail.com, ozturaca@boun.edu.tr if necessary
 
 *  Used backfill plugin as base code. There are some functions that are
    left untouched.

 *  For the theory behind the plugin, see the manuscript
 *  doi: ....
\*****************************************************************************/

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "slurm/slurm.h"
#include "slurm/slurm_errno.h"

#include "src/common/list.h"
#include "src/common/macros.h"
#include "src/common/node_select.h"
#include "src/common/parse_time.h"
#include "src/common/slurm_protocol_api.h"
#include "src/common/xmalloc.h"
#include "src/common/xstring.h"

#include "src/slurmctld/acct_policy.h"
#include "src/slurmctld/front_end.h"
#include "src/slurmctld/job_scheduler.h"
#include "src/slurmctld/licenses.h"
#include "src/slurmctld/locks.h"
#include "src/slurmctld/node_scheduler.h"
#include "src/slurmctld/preempt.h"
#include "src/slurmctld/reservation.h"
#include "src/slurmctld/slurmctld.h"
#include "src/slurmctld/srun_comm.h"
#include "src/plugins/select/lpconsres/select_lpconsres.h"
#include "aucsched.h"

/* run every 5 seconds */
#ifndef SCHED_INTERVAL
#  define SCHED_INTERVAL	5
#endif

/* max # of jobs = 20 */
/* max # of bids = 15000 */
#ifndef MAX_JOB_COUNT
#define   MAX_JOB_COUNT 50
#endif

#ifndef MAX_BID_COUNT
#define   MAX_BID_COUNT 150000
#endif

#define SLURMCTLD_THREAD_LIMIT	5

extern int gres_job_gpu_count(List job_gres_list, int *gpu, int *gpu_max);
static char cplex_license_address[256];
extern int solve_allocation(int m, int n, int timeout, 
			sched_nodeinfo_t *node_array, 
			solver_job_list_t *job_array,
			int max_bid_count);
/*********************** local variables *********************/
static bool stop_aucsched = false;
static pthread_mutex_t term_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  term_cond = PTHREAD_COND_INITIALIZER;
static bool config_flag = false;
static uint32_t debug_flags = 0;
static int sched_interval = SCHED_INTERVAL;
static int max_job_count = MAX_JOB_COUNT;
static int max_bid_count = MAX_BID_COUNT;

/*********************** local functions *********************/
static int  _run_solver_opt(void);
static void _load_config(void);
static int  _start_job(struct job_record *job_ptr, bitstr_t *bitmap, bitstr_t *avail_bitmap);
extern struct node_schedinfo *_return_nodes();
extern int print_nodes_lpconsres(struct node_record *node_ptr, int node_cnt);

/* Terminate aucsched_agent */
extern void stop_aucsched_agent(void)
{
	pthread_mutex_lock(&term_lock);
	stop_aucsched = true;
	pthread_cond_signal(&term_cond);
	pthread_mutex_unlock(&term_lock);
}

static void _load_config(void)
{
	char *sched_params, *tmp_ptr;

	sched_params = slurm_get_sched_params();
	debug_flags  = slurm_get_debug_flags();

	if (sched_params && (tmp_ptr=strstr(sched_params, "interval=")))
		sched_interval = atoi(tmp_ptr + 9);
	if (sched_interval < 1) {
		fatal("Invalid scheduler interval: %d",
		      sched_interval);
	}
	if (sched_params && (tmp_ptr=strstr(sched_params, "cplex_lic="))) {
		sprintf(cplex_license_address,"%s",tmp_ptr + 10);
	}

	if (sched_params && (tmp_ptr=strstr(sched_params, "max_job_count=")))
		max_job_count = atoi(tmp_ptr + 14);
	if (max_job_count < 1) {
		fatal("Invalid aucsched max_job_count: %d",
		      max_job_count);
	}
	xfree(sched_params);
}

extern char* get_cplex_license_address(void)
{
	debug("cplex: %s",cplex_license_address);	
	return cplex_license_address;
}

/* Note that slurm.conf has changed */
extern void aucsched_reconfig(void)
{
	config_flag = true;
}

extern void *aucsched_agent(void *args)
{
	DEF_TIMERS;
	time_t now;
	double wait_time;
	static time_t last_aucsched_time = 0;
	/* Read config and partitions; Write jobs and nodes */
//	slurmctld_lock_t all_locks = {
//		READ_LOCK, WRITE_LOCK, WRITE_LOCK, READ_LOCK };

	_load_config();
	last_aucsched_time = time(NULL);
	while (!stop_aucsched) {
		//_my_sleep(sched_interval);
		if (stop_aucsched)
			break;
		if (config_flag) {
			config_flag = false;
			_load_config();
		}
		now = time(NULL);
		wait_time = difftime(now, last_aucsched_time);
		//wait_time = sched_interval; // temporary
		if (!avail_front_end() || (wait_time < sched_interval))
			continue;
		START_TIMER;
		//lock_slurmctld(all_locks);
		while (_run_solver_opt()) ;
		debug3("left runsolveropt");
		//unlock_slurmctld(all_locks);
		last_aucsched_time = time(NULL);
		END_TIMER;
	}
	return NULL;
}

/* Try to start the job on any non-reserved nodes */
static int _start_job(struct job_record *job_ptr, bitstr_t *bitmap,
	bitstr_t *avail_bitmap)
{
	int rc;
	bitstr_t *orig_exc_nodes = NULL;
	static uint32_t fail_jobid = 0;

	bit_not(bitmap);
	job_ptr->details->exc_node_bitmap = bit_copy(bitmap);
	rc = select_nodes(job_ptr, false, NULL);
	if (job_ptr->details) { /* select_nodes() might cancel the job! */
		FREE_NULL_BITMAP(job_ptr->details->exc_node_bitmap);
		job_ptr->details->exc_node_bitmap = orig_exc_nodes;
	} else
		FREE_NULL_BITMAP(orig_exc_nodes);
	if (rc == SLURM_SUCCESS) {
		/* job initiated */
		last_job_update = time(NULL);
		info("aucsched: Started JobId=%u on %s",
		     job_ptr->job_id, job_ptr->nodes);
		if (job_ptr->batch_flag == 0)
			srun_allocate(job_ptr->job_id);
		else if (job_ptr->details->prolog_running == 0)
			launch_job(job_ptr);
	} else if ((job_ptr->job_id != fail_jobid) &&
		   (rc != ESLURM_ACCOUNTING_POLICY)) {
		char *node_list;
		node_list = bitmap2node_name(bitmap);
		/* This happens when a job has sharing disabled and
		 * a selected node is still completing some job,
		 * which should be a temporary situation. */
		verbose("aucsched: Failed to start JobId=%u on %s: %s",
			job_ptr->job_id, node_list, slurm_strerror(rc));
		xfree(node_list);
		fail_jobid = job_ptr->job_id;
		debug3("node state NODE_STATE_DRAIN %d",NODE_STATE_DRAIN);
		debug3("node state NODE_STATE_COMPLETING %d",NODE_STATE_COMPLETING);
		debug3("node state NODE_STATE_NO_RESPOND %d",NODE_STATE_NO_RESPOND);
		debug3("node state NODE_STATE_POWER_SAVE %d",NODE_STATE_POWER_SAVE);
		debug3("node state NODE_STATE_POWER_UP %d",NODE_STATE_POWER_UP);
		debug3("node state NODE_STATE_FAIL %d",NODE_STATE_FAIL);
		debug3("node state NODE_STATE_DRAIN %d",NODE_STATE_POWER_UP);
		debug3("node state NODE_STATE_DRAIN %d",NODE_STATE_MAINT);
		debug3("node state NODE_STATE_DRAIN %d",NODE_STATE_IDLE);
		debug3("node state NODE_STATE_DRAIN %d",NODE_STATE_UNKNOWN);
		debug3("node state NODE_STATE_DRAIN %d",NODE_STATE_DOWN);
		debug3("node state NODE_STATE_DRAIN %d",NODE_STATE_ALLOCATED);
		debug3("node state NODE_STATE_DRAIN %d",NODE_STATE_ERROR);
		debug3("node state NODE_STATE_DRAIN %d",NODE_STATE_MIXED);
	} else {
		debug3("aucsched: Failed to start JobId=%u: %s",
		       job_ptr->job_id, slurm_strerror(rc));
	}

	return rc;
}

static void free_and_null (char **ptr)
{
	if ( *ptr != NULL ) {
		free (*ptr);
		*ptr = NULL;
	}
}


static int _run_solver_opt(void)
{
	bool filter_root = false;
	sched_nodeinfo_t* node_array = NULL;
	List job_queue;
	job_queue_rec_t *job_queue_rec;
	slurmdb_qos_rec_t *qos_ptr = NULL;
	int i, j, solver_job_idx;
	struct job_record *job_ptr;
	solver_job_list_t *job_list, *sjob_ptr;
	struct part_record *part_ptr = NULL;
	uint32_t end_time, time_limit, comp_time_limit, orig_time_limit;
	uint32_t min_nodes, max_nodes, req_nodes;
	bitstr_t *avail_bitmap = NULL, *resv_bitmap = NULL;
        bitstr_t *exc_core_bitmap = NULL;
	time_t now = time(NULL), sched_start, later_start, start_res;
	static int sched_timeout = 0, solution_status;
	int this_sched_timeout = 0, rc = 0, job_list_size;
	sched_start = now;
	if (sched_timeout == 0) {
		sched_timeout = slurm_get_msg_timeout() / 2;
		sched_timeout = MAX(sched_timeout, 1);
		sched_timeout = MIN(sched_timeout, 10);
	}
	this_sched_timeout = sched_timeout;

	if (slurm_get_root_filter())
		filter_root = true;

	job_queue = build_job_queue(true);
	if (list_count(job_queue) < 1) {
		debug("sched: no jobs to run the solver");
		list_destroy(job_queue);
		return 0;
	}

	debug("generating job queue at %lld",(long long)sched_start);
	debug("queue size: %d, max_job_count: %d",list_count(job_queue),max_job_count);
	debug("max bid count: %d",max_bid_count);
	/* generate job queue */
	job_list_size = MIN(max_job_count, list_count(job_queue));
	job_list = (solver_job_list_t*)malloc(
			sizeof(solver_job_list_t) * job_list_size);
	/* will be converted to priority ordered list */
	solver_job_idx = 0;
	while ((job_queue_rec = (job_queue_rec_t *)
				list_pop_bottom(job_queue, sort_job_queue2))) {
		job_ptr  = job_queue_rec->job_ptr;
		part_ptr = job_queue_rec->part_ptr;
		xfree(job_queue_rec);
		if (!IS_JOB_PENDING(job_ptr))
			continue;	/* started in other partition */
		job_ptr->part_ptr = part_ptr;

		if ((job_ptr->state_reason == WAIT_ASSOC_JOB_LIMIT) ||
		    (job_ptr->state_reason == WAIT_ASSOC_RESOURCE_LIMIT) ||
		    (job_ptr->state_reason == WAIT_ASSOC_TIME_LIMIT) ||
		    (job_ptr->state_reason == WAIT_QOS_JOB_LIMIT) ||
		    (job_ptr->state_reason == WAIT_QOS_RESOURCE_LIMIT) ||
		    (job_ptr->state_reason == WAIT_QOS_TIME_LIMIT) ||
		    !acct_policy_job_runnable(job_ptr)) {
			debug2("aucsched: job %u is not allowed to run now. "
			       "Skipping it. State=%s. Reason=%s. Priority=%u",
			       job_ptr->job_id,
			       job_state_string(job_ptr->job_state),
			       job_reason_string(job_ptr->state_reason),
			       job_ptr->priority);
			continue;
		}

		if (((part_ptr->state_up & PARTITION_SCHED) == 0) ||
		    (part_ptr->node_bitmap == NULL))
		 	continue;
		if ((part_ptr->flags & PART_FLAG_ROOT_ONLY) && filter_root)
			continue;

		if ((!job_independent(job_ptr, 0)) ||
		    (license_job_test(job_ptr, time(NULL)) != SLURM_SUCCESS))
			continue;

		/* Determine minimum and maximum node counts */
		min_nodes = MAX(job_ptr->details->min_nodes,
				part_ptr->min_nodes);
		if (job_ptr->details->max_nodes == 0)
			max_nodes = part_ptr->max_nodes;
		else
			max_nodes = MIN(job_ptr->details->max_nodes,
					part_ptr->max_nodes);
		max_nodes = MIN(max_nodes, 500000);     /* prevent overflows */
		if (job_ptr->details->max_nodes)
			req_nodes = max_nodes;
		else
			req_nodes = min_nodes;
		if (min_nodes > max_nodes) {
			/* job's min_nodes exceeds partition's max_nodes */
			continue;
		}
		/* Determine job's expected completion time */
		if (job_ptr->time_limit == NO_VAL) {
			if (part_ptr->max_time == INFINITE)
				time_limit = 365 * 24 * 60; /* one year */
			else
				time_limit = part_ptr->max_time;
		} else {
			if (part_ptr->max_time == INFINITE)
				time_limit = job_ptr->time_limit;
			else
				time_limit = MIN(job_ptr->time_limit,
						 part_ptr->max_time);
		}
		comp_time_limit = time_limit;
		orig_time_limit = job_ptr->time_limit;
		if (qos_ptr && (qos_ptr->flags & QOS_FLAG_NO_RESERVE))
			time_limit = job_ptr->time_limit = 1;
		else if (job_ptr->time_min && (job_ptr->time_min < time_limit))
			time_limit = job_ptr->time_limit = job_ptr->time_min;

		/* Determine impact of any resource reservations */
		later_start = now;
 		FREE_NULL_BITMAP(avail_bitmap);
		FREE_NULL_BITMAP(exc_core_bitmap);
		start_res   = later_start;
		later_start = 0;
		j = job_test_resv(job_ptr, &start_res, true, &avail_bitmap,
				  &exc_core_bitmap);
		if (j != SLURM_SUCCESS) {
			job_ptr->time_limit = orig_time_limit;
			continue;
		}
		if (start_res > now)
			end_time = (time_limit * 60) + start_res;
		else
			end_time = (time_limit * 60) + now;

		sjob_ptr = &job_list[solver_job_idx++];
		sjob_ptr->job_ptr = job_ptr;
		sjob_ptr->job_id = job_ptr->job_id;
		debug3("before decision min_nodes: %d max_nodes: %d",min_nodes,max_nodes);
		if ( (min_nodes == 1) && (max_nodes == 500000) ) {
			sjob_ptr->min_nodes = 0;
			sjob_ptr->max_nodes = 500000;
			debug3("job %d min_nodes: %d, max_nodes: %d, node count disabled.",
				sjob_ptr->job_id, min_nodes, max_nodes);
		} else {
			sjob_ptr->min_nodes = min_nodes;
			sjob_ptr->max_nodes = max_nodes;
			debug3("job %d min_nodes: %d, max_nodes: %d, node count set to %d.",
				sjob_ptr->job_id, min_nodes, max_nodes, sjob_ptr->min_nodes);
		}
		sjob_ptr->contiguous = job_ptr->details->contiguous;
		gres_job_gpu_count(job_ptr->gres_list, &sjob_ptr->gpu, &sjob_ptr->gpu_max);
		if (sjob_ptr->gpu_max == 0) {
			sjob_ptr->gpu_max = sjob_ptr->gpu;
		}
		if (sjob_ptr->gpu == 0) {
			sjob_ptr->gpu_max = 0;
		}
		sjob_ptr->min_cpus = job_ptr->details->min_cpus;
		sjob_ptr->cpus_per_node = job_ptr->details->ntasks_per_node *
						job_ptr->details->cpus_per_task;
		sjob_ptr->exclusive = (job_ptr->details->shared == 0);
		debug2("job exclusive flag: %d shared flag: %d",sjob_ptr->exclusive, job_ptr->details->shared);

		if (sjob_ptr->cpus_per_node > 0) {
			sjob_ptr->max_cpus = MIN(job_ptr->details->max_cpus,sjob_ptr->cpus_per_node * sjob_ptr->max_nodes);
		} else {
			sjob_ptr->max_cpus = sjob_ptr->min_cpus;
		}

		sjob_ptr->priority = job_ptr->priority;

		debug("job %d id: %d min_cpus: %d max_cpus: %u cpus_per_node %d min_nodes: %d max_nodes: %d gpu: %d, gpu_max: %d, prio: %d",
			solver_job_idx-1, sjob_ptr->job_id, sjob_ptr->min_cpus, sjob_ptr->max_cpus,
			sjob_ptr->cpus_per_node, sjob_ptr->min_nodes, sjob_ptr->max_nodes, sjob_ptr->gpu,
			sjob_ptr->gpu_max, sjob_ptr->priority);
		if (solver_job_idx >= max_job_count)
			break;
	}
	debug("job_list_size: %d",job_list_size);
	node_array = _print_nodes_inaucsched(avail_bitmap,cg_node_bitmap);
	debug3("before allocation, heres the result:");
	
	for (i=0;i<node_record_count;i++)
		debug3("aucschedde node %d remgpu %u remcpu %u",
			i,node_array[i].rem_gpus,node_array[i].rem_cpus);

	uint32_t diffPrio = 1 - job_list[job_list_size - 1].priority;
	for (i = 0; i < job_list_size; i++) {
		job_list[i].priority += diffPrio;
	}
	
	solution_status = solve_allocation(node_record_count, job_list_size, 
		sched_interval, node_array, job_list, max_bid_count);
	if (solution_status == 2) {
		max_bid_count = (int)(max_bid_count/1.5);
		return 0;
	}
	else if (!solution_status) {
		max_bid_count = (int)(max_bid_count*1.5);
		if (max_bid_count > MAX_BID_COUNT)
			max_bid_count = MAX_BID_COUNT;
	} else // if (solution_status)
		return 0;
	/*	
	debug3("xafter allocation, heres the result:");
	for (i=0;i<node_record_count;i++)
		debug3("node %d remgpu %u remcpu %u",
			i,node_array[i].rem_gpus,node_array[i].rem_cpus);
	*/
	for (i=0;i<job_list_size;i++) {
		sjob_ptr = &job_list[i];
		job_ptr = sjob_ptr->job_ptr;
		if (!sjob_ptr->alloc_total) { 
			debug3("job %d not allocated, continuing.",sjob_ptr->job_id);
			continue;
		} 
		debug3("job %d min_cpus: %d, max_cpus: %d allocated: %d",
			sjob_ptr->job_id, sjob_ptr->min_cpus, sjob_ptr->max_cpus, sjob_ptr->alloc_total);
		bit_and(avail_bitmap, sjob_ptr->node_bitmap); 		
		/* Identify usable nodes for this job */
		bit_and(avail_bitmap, part_ptr->node_bitmap);
		bit_and(avail_bitmap, up_node_bitmap);

		if (job_ptr->details->exc_node_bitmap) {
			bit_not(job_ptr->details->exc_node_bitmap);
			bit_and(avail_bitmap,
				job_ptr->details->exc_node_bitmap);
			bit_not(job_ptr->details->exc_node_bitmap);
		}

		/* Identify nodes which are definitely off limits */
		FREE_NULL_BITMAP(resv_bitmap);
		resv_bitmap = bit_copy(avail_bitmap);
		bit_not(resv_bitmap);

/*		if ((time(NULL) - sched_start) >= this_sched_timeout) {
			debug("aucsched: loop taking too long, yielding locks");
			if (_yield_locks()) {
				debug("aucsched: system state changed, "
				      "breaking out");
				rc = 1;
				break;
			} else {
				this_sched_timeout += sched_timeout;
			}
		}
*/
		rc = _start_job(job_ptr, sjob_ptr->node_bitmap, avail_bitmap);
		if (rc != SLURM_SUCCESS) {
			debug("Could not start JobId %d, one of the nodes busy",job_ptr->job_id);
/*			for (i=0;i<node_record_count;i++)
				debug3("during startjob node %d remcpu %u job onnode %u state %d",
					i,node_array[i].rem_cpus,job_ptr->details->req_node_layout[i],
					node_record_table_ptr[i].node_state);
			fatal("exiting, error: %d %s", rc, slurm_strerror(rc));	
*/
		}
	}

	free_and_null((char **) &job_list);
	FREE_NULL_BITMAP(avail_bitmap);
	FREE_NULL_BITMAP(resv_bitmap);

	list_destroy(job_queue);
	return rc;
}

/*
takes the empty cpu & gpu information from the select plugin
and returns it to the ilp solver
*/
struct sched_nodeinfo* _print_nodes_inaucsched(bitstr_t *avail_bitmap,
	bitstr_t *cg_node_bitmap)
{
	int i;
	struct select_nodeinfo *nodeinfo = NULL;
	struct sched_nodeinfo* node_array = xmalloc(
		sizeof(struct sched_nodeinfo* ) * node_record_count);
	select_g_select_nodeinfo_set_all();
	for (i = 0; i < node_record_count; i++) {
		if (bit_test(avail_bitmap, i)) {
			select_g_select_nodeinfo_get(
				node_record_table_ptr[i].select_nodeinfo, 
				SELECT_NODEDATA_PTR, 0, (void *)&nodeinfo);
			if(!nodeinfo) {
				error("no nodeinfo returned from structure");
				continue;
			}
			node_array[i].rem_gpus = nodeinfo->rem_gpus;
			node_array[i].rem_cpus = nodeinfo->rem_cpus;
			node_array[i].allocated = (nodeinfo->alloc_cpus > 0);
			/*
			info("node %d cpu %u",i,node_record_table_ptr[i].cpus);
			info("aucschedde alloc cpu in node %d is %u, rem_gpu %u",
				i,nodeinfo->alloc_cpus,nodeinfo->rem_gpus); 
			*/
			
		} else {
			node_array[i].rem_cpus = 0;
			node_array[i].allocated = true;
			node_array[i].rem_gpus = 0;
		}
	}
	return node_array;
}


