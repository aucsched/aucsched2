#include <solver_cplex.h>
#include <math.h>

#define MAXBIDPERJOB 15
#define MAXBIDS 250
#define MAXJOBNODE 5120000

#define MAXCLASSESPERJOB 500
#define MAXCLASSES 100000
#define MAXINTERVALS 1000000

#define NONCONTIGLEVEL 100

#define LARGENUMBER 10000000

int classCtr, classMapCtr, jobNodeCtr, intervalCtr;
int maxBids;

static void free_and_null (char **ptr);

extern char* get_cplex_license_address(void);

int bidCtr;
int max_gpu;

uint32_t minPrioDiff;

typedef struct bid_t {
	int class;
	int jobIdx;
	int amount;
	int numNodes;
	int firstNode;
	int lastNode;
	double priority;
	double preference;
} bid;

typedef struct class_map_t {
	int nodes;
	int cpus;
	int gpus;
	int begin;
	int end;
} class_map;

typedef struct interval_t {
	int begin;
	int end;
	int size;
	int numnodes;
	//QQint *nodes;
	int gpu;
} interval;

typedef struct class_t {
	int numCores;
	int numNodes;
	int numGpus;
	int *nodesUsed;
	double preference;
	/* sil */
	struct class_bidding_job_t *bidding_job;
	int gpn;
	int cpn;
	int amount;
	int sumBids;
	bool isAligned;
} class;

/* sil */
typedef struct class_type_t {
	int gpn;
	int cpn;
	struct class_t *first;
} class_type;

/* sil */
typedef struct nodemap_t {
	int begin;
	int end;
	struct nodemap_t *next;
} nodemap;

/* holds the LAST element's index for a specific CPN/GPN class exists */
int *cidxArray;
/* same for class union list */
int *cuidxArray;
/* same for class-node list */
int *nidxArray;
int *nArray; 

double alpha;

int *jobNodeIdx, *jobNodeName;
class_map *classMap;
interval *intervals;
int *nodesetCount;
int *cumsum;

class *cArray;
class *cuArray;

bid *bids;

//slurmBids
char *avail_nodelist, *nodelist;
sched_nodeinfo_t* temp_node_array;

//solver_job_list_t *job_array;
//sched_nodeinfo_t *node_array;

/* generate bid for a specific job and class */


double preferenceCalculation(int numNodes, int nodeSize, int numNodesets, int nodesetSize, double c1, double c2, double c3) {
        double ret = 1.0;
        ret -= (c1 + c2 * numNodes / (1 + nodeSize) + c3 * numNodesets / (1 + nodesetSize));
        debug4("c1 %.2f c2 %.2f c3 %.2f numNodes %d nodeSize %d numNodesets %d nodesetSize %d, returning %.2f",
                c1, c2, c3, numNodes, nodeSize, numNodesets, nodesetSize, ret);
        return ret;
}

int newBid(int amount, int jobIdx, int cIdx, double priority, double preference, int numNodes, int *_nodes, solver_job_list_t *job_array)
{	
        if (bidCtr < maxBids) {
	        int i, ok = 1;
        	for (i = job_array[jobIdx].firstBid; i < bidCtr; i++) {
	                if (cIdx == bids[i].class) {
	                        debug3("job %d has already a bid on class %d, skipping",
	                                job_array[jobIdx].job_id, cIdx);
                                return 0;
                        }
                }
        	if (ok) {
                debug3("creating bid for job %d, class %d, numnodes %d, last of nodes %d",
                        jobIdx, cIdx, numNodes, _nodes[numNodes - 1]);
        	bids[bidCtr].class = cIdx;
        	bids[bidCtr].amount = amount;
        	bids[bidCtr].jobIdx = jobIdx;
        	bids[bidCtr].priority = priority;
        	bids[bidCtr].preference = preference;
        	bids[bidCtr].numNodes = numNodes;
        	bids[bidCtr].firstNode = jobNodeCtr;
        	for (i = 0; i < numNodes; i++) {
	                debug3("job %d bidding on node %d (%d) bid id %d",jobIdx,_nodes[i],jobNodeName[jobNodeCtr],bidCtr+1);
	                jobNodeName[jobNodeCtr] = _nodes[i];
                	jobNodeIdx[jobNodeCtr] = jobNodeCtr;
	        	jobNodeCtr++;
                	nArray[_nodes[i]]++;
        	}
        	bids[bidCtr].lastNode = jobNodeCtr;
        	debug("created bid no %d, class %d, job %d, numNodes %d firstNode %d lastNode %d, prio: %.2f, pref: %.2f",
                      bidCtr, cIdx, jobIdx, numNodes, bids[bidCtr].firstNode, bids[bidCtr].lastNode, priority, preference);
                      bidCtr++;
        	if (jobNodeCtr >= MAXJOBNODE)
	              maxBids = bidCtr;
	        }
	        cArray[cIdx].preference = 0.00;
	        return 1;
        } else {
                return -1;
        }
}

void cumSumGeneration(int nodeSize, sched_nodeinfo_t *node_array)
{
	int g, i, k;
	for (i = 0; i < nodeSize; i++) {
		if (node_array[i].rem_gpus > max_gpu)
			max_gpu = node_array[i].rem_gpus;
	}
	
	nodesetCount = (int *)malloc((max_gpu + 1) * sizeof(int));
	cumsum = (int *)malloc((1 + nodeSize) * (1 + max_gpu) * sizeof(int));
	debug("cumsum created. max_gpu: %d", max_gpu);
	
	intervalCtr = 0;
	for (g = 0; g <= max_gpu; g++) {
	        nodesetCount[g] = 0;
	        cumsum[g * (nodeSize + 1)] = 0;
		for (i = 1; i <= nodeSize; i++) {
			if (node_array[i - 1].rem_gpus >= g)
				cumsum[g * (nodeSize + 1) + i] = cumsum[g * (nodeSize + 1) + i - 1] + node_array[i - 1].rem_cpus;
			else
				cumsum[g * (nodeSize + 1) + i] = cumsum[g * (nodeSize + 1) + i - 1];
		}
		
		debug3("last element of cumsum with gpu %d is %d",g,cumsum[(g + 1) * (nodeSize + 1) - 1]);
		
		k = g * (nodeSize + 1); // k is starting point
		while (k < (g + 1) * (nodeSize + 1)) {
		        // find the beginning point 'k', where 'k+1' has higher cumsum then k.
                        if ( (k <= (g + 1) * (nodeSize + 1)) && (cumsum[k + 1] - cumsum[k] == 0) ) {
                                k++;
                                continue;
                        }
	 		for (i = k; i < (g + 1) * (nodeSize + 1); i++) {
                        	if ( (cumsum[i + 1] - cumsum[i] <= 0) || 
                        	     ( (i + 1 == (g + 1) * (nodeSize + 1)) && (cumsum[i] - cumsum[i - 1] > 0 ) ) ) {
                        	        intervals[intervalCtr].begin = k - g * (nodeSize + 1);
                                	intervals[intervalCtr].end = i - g * (nodeSize + 1) - 1;
                                	intervals[intervalCtr].numnodes = i - k;
					//intervals[intervalCtr].nodes = (int *)malloc(intervals[intervalCtr].numnodes * sizeof(int));
					//for (p = 0; p < intervals[intervalCtr].numnodes; p++) {
						//intervals[intervalCtr].nodes[p] = intervals[intervalCtr].begin + p;
					        //debug3("interval %d node %d: %d",intervalCtr,p,intervals[intervalCtr].nodes[p]);
                                        //}
                                	intervals[intervalCtr].size = cumsum[i] - cumsum[k];
                                	intervals[intervalCtr].gpu = g;
                                	debug3("new interval 1, intervalCtr %d, begin %d , end %d",
                                                intervalCtr, intervals[intervalCtr].begin, intervals[intervalCtr].end);
                                	k = i + 1; 
                                	intervalCtr++;
                                	nodesetCount[g]++;
                                	break;
                        	}
	 		}
	 		k++;
        	}
        }
        
        /* interval debug lines */
        for (i = 0; i < intervalCtr; i++) {
                debug3("interval id: %d, begin: %d, end: %d, size: %d, numnodes: %d, gpu: %d",
                        i, intervals[i].begin, intervals[i].end, intervals[i].size,
                        intervals[i].numnodes, intervals[i].gpu);
        }
        
	/* debug lines */
	//for (i = 0; i < (nodeSize + 1) * (1 + max_gpu); i++) {
	//        debug3("i %d, cumsum %d", i, cumsum[i]);
	//}
      
}

// slurm bids
/*
void slurmBids(solver_job_list_t *job_array, int nodeSize, int windowSize, sched_nodeinfo_t *node_array) {
        int job_idx;
        int i; // interval idx
        int ret; // dummy to hold return value from select_p_job_test
        bitstr_t *avail_bitmap = bit_alloc (nodeSize);
        bitstr_t *bitmap = bit_alloc (nodeSize);
        for (i = 0; i < nodeSize; i++) {
                bit_set (bitmap, (bitoff_t) (i));
                bit_set (avail_bitmap, (bitoff_t) (i));
        }
        char *avail_nodelist, *nodelist;
        nodelist = bitmap2node_name(bitmap);
        avail_nodelist = bitmap2node_name(avail_bitmap);
        debug("in slurmBids; avail_bitmap: %s, bitmap: %s", avail_nodelist, nodelist);
                                
        for (job_idx = 0; job_idx < windowSize; job_idx++) {                                        
                for (i = 0; i < intervalCtr; i++) {
                        if ( (intervals[i].gpu == job_array[job_idx].gpu) &&
                        	(intervals[i].size >= job_array[job_idx].min_cpus) &&
                        	(intervals[i].numnodes >= job_array[job_idx].min_nodes) ) {
                                ret = select_g_job_test(job_array[job_idx].job_ptr, bitmap,
                                        job_array[job_idx].min_nodes, job_array[job_idx].max_nodes,
                                        job_array[job_idx].min_nodes, SELECT_MODE_WILL_RUN,
                                        NULL, NULL);
                                if (ret == 0) {
                                        nodelist = bitmap2node_name(bitmap);
                                        debug3("in slurmBids; job_test for job %d returned %s", job_idx, nodelist);
                                        // buraya bid olusturmak gerekli, bitmap'teki nodelara gore.
                                        // sonra da avail bitmap update
                                } else {
                                        debug3("in slurmBids; job_test for job %d returned false value %d", job_idx, ret);
                                }
                        }
                }
        }
}
*/

// class generation for node-only jobs
void classGenerationN(solver_job_list_t *job_array, int job_idx, int nodeSize, sched_nodeinfo_t *node_array) 
{ 
        int i, j, k, l, cores, m, p, end_node, ret;
        int firstContigClass, firstClass, lastClass, lastContClass, lastAlignedClass, numClass;
        bool ok;
	debug3("entering classmap loop, classmapctr %d",classMapCtr);
        interval temp;
        int *nodes = (int *)malloc(nodeSize * sizeof(int));

	firstClass = classCtr;
        job_array[job_idx].firstBid = bidCtr;
        // slurmBids
/*
        bitstr_t *bitmap = bit_alloc (nodeSize);
        for (i = 0; i < nodeSize; i++) {
                if (temp_node_array[i].rem_cpus > 0) {
                        debug3("temp node array node %d remcpu %d",i,temp_node_array[i].rem_cpus);
                        bit_set (bitmap, (bitoff_t) (i));
                }
        }
        nodelist = bitmap2node_name(bitmap);
        debug3("in slurmBids; input to job_test for job %d (%d) input %s",
                job_array[job_idx].job_id, job_idx, nodelist);

        ret = select_g_job_test(job_array[job_idx].job_ptr, bitmap,
                job_array[job_idx].min_nodes, job_array[job_idx].max_nodes,
                job_array[job_idx].min_nodes, SELECT_MODE_WILL_RUN,
                NULL, NULL);
        if (ret == 0) {
                nodelist = bitmap2node_name(bitmap);
                debug3("in slurmBids; job_test for job %d (%d)returned %s",
                        job_array[job_idx].job_id, job_idx, nodelist);
                // buraya bid olusturmak gerekli, bitmap'teki nodelara gore.
		int sum = 0;
                for (i = 0; i < nodeSize; i++) {
                        if (bit_test(bitmap, i)) {
                                sum += temp_node_array[i].rem_cpus;
                        }
                }
		if (sum >= job_array[job_idx].min_cpus) {
                int numNodes = bit_set_count(bitmap);
                int *nodes = (int*)malloc(numNodes * sizeof(int));
                int j = 0;
                int contig = 0;
                for (i = 0; i < nodeSize; i++) {
                        if (bit_test(bitmap, i)) {
                                nodes[j] = i;
                                j++;
                        }
                }
                for (i = 1; i < j; i++) {
                        if (nodes[i] - nodes[i-1] > 1)
                                contig++;
                }
                debug3("in slurmBids; job_test for job %d is on %d contig parts", job_array[job_idx].job_id, contig);
                
                cArray[classCtr].numNodes = numNodes;
                cArray[classCtr].nodesUsed = (int *)malloc(cArray[classCtr].numNodes * sizeof(int));
                int classIdx = classCtr++;
                
                double priority = job_array[job_idx].priority;
                int amount = MIN(job_array[job_idx].max_cpus, sum);
                double preference = preferenceCalculation
                        (numNodes, nodeSize, contig,
                        nodesetCount[job_array[job_idx].gpu], 0.5, 0.0, 0.5);
                newBid(amount, job_idx, classIdx, priority, preference, numNodes, nodes, job_array);
                // sonra da temp_node_array update
                for (j = 0; j < numNodes; j++) {
                        k = nodes[j];
                        if (job_array[job_idx].cpus_per_node > 0) {
                                temp_node_array[k].rem_cpus -= job_array[job_idx].cpus_per_node;
                        } else {
                                temp_node_array[k].rem_cpus = 0;
                        }
                }
		}
        } else {
                debug3("in slurmBids; job_test for job %d returned false value %d", job_idx, ret);
        }
*/
        firstContigClass = classCtr;
	/* contiguous aligned bids */
	for (i = 0; i < intervalCtr; i++) {
	        if (classCtr >= MAXCLASSES) {
			debug3("max number of classes (%d) reached",MAXCLASSES);
			break;
		}		       
                if ( (intervals[i].gpu == job_array[job_idx].gpu) &&
                	(intervals[i].size >= job_array[job_idx].min_cpus) &&
                	(intervals[i].numnodes >= job_array[job_idx].min_nodes) ) {
                        int offset = job_array[job_idx].gpu * (nodeSize + 1);
                        // align to beginning of interval
                        k = intervals[i].begin;
               		end_node = MIN(job_array[job_idx].max_nodes, intervals[i].end - k + 1);
                       	cores = cumsum[k + end_node + offset] - cumsum[k + offset];
                       	debug3("begin node %d end_node %d cores %d ",k,end_node,cores);
                        if ( (cores >= job_array[job_idx].min_cpus) ) {
                                ok = true;
                                if (job_array[job_idx].cpus_per_node > 0)
                                        for (j = k; j < k + end_node; j++) {
                                                if (node_array[j].rem_cpus < job_array[job_idx].cpus_per_node) {
                                                        ok = false;
                                                }
                                        }
                                if (ok) {
                                        debug3("creating class for cores %d",cores);
                                	cArray[classCtr].numNodes = end_node;
                                        if (job_array[job_idx].cpus_per_node > 0)
                                        	cArray[classCtr].numCores = end_node * job_array[job_idx].cpus_per_node;
                                        else
                                                cArray[classCtr].numCores = MIN(cores, job_array[job_idx].max_cpus);
                                        cArray[classCtr].numGpus = job_array[job_idx].gpu;
                                        cArray[classCtr].nodesUsed = (int *)malloc(cArray[classCtr].numNodes * sizeof(int));
                                        cArray[classCtr].sumBids = 0;
					cArray[classCtr].preference = preferenceCalculation
					        (cArray[classCtr].numNodes, nodeSize, 1, 
					        nodesetCount[cArray[classCtr].numGpus], 0.0, 0.0, 0.0);
                                        for (m = 0; m < cArray[classCtr].numNodes; m++) {
                                        	cArray[classCtr].nodesUsed[m] = k + m;
                                        	debug3("nodeUsed %d: %d",m,cArray[classCtr].nodesUsed[m]);
                                        }
                                       	classCtr++;
                                }
                        }
                        // align to end of interval
                        k = intervals[i].end - job_array[job_idx].min_nodes + 1;
               		end_node = MIN(job_array[job_idx].max_nodes, intervals[i].end - k + 1);
                       	cores = cumsum[k + end_node + offset] - cumsum[k + offset];
                       	debug3("begin node %d end_node %d cores %d ",k,end_node,cores);
                        if ( (cores >= job_array[job_idx].min_cpus) ) {
                                ok = true;
                                if (job_array[job_idx].cpus_per_node > 0)
                                        for (j = k; j < k + end_node; j++) {
                                                if (node_array[j].rem_cpus < job_array[job_idx].cpus_per_node) {
                                                        ok = false;
                                                }
                                        }
                                if (ok) {
                                        debug3("creating class for cores %d",cores);
                                	cArray[classCtr].numNodes = end_node;
                                        if (job_array[job_idx].cpus_per_node > 0)
                                        	cArray[classCtr].numCores = end_node * job_array[job_idx].cpus_per_node;
                                        else
                                                cArray[classCtr].numCores = MIN(cores, job_array[job_idx].max_cpus);
                                        cArray[classCtr].numGpus = job_array[job_idx].gpu;
                                        cArray[classCtr].nodesUsed = (int *)malloc(cArray[classCtr].numNodes * sizeof(int));
                                        cArray[classCtr].sumBids = 0;
                                        cArray[classCtr].preference = preferenceCalculation
                                                (cArray[classCtr].numNodes, nodeSize, 1,
                                                nodesetCount[cArray[classCtr].numGpus], 0.0, 0.0, 0.0);
                                        for (m = 0; m < cArray[classCtr].numNodes; m++) {
                                        	cArray[classCtr].nodesUsed[m] = k + m;
                                        	debug3("nodeUsed %d: %d",m,cArray[classCtr].nodesUsed[m]);
                                        }
                                       	classCtr++;
                                }
			}

		}
	}
	debug3("created %d aligned contig classes for job %d",classCtr - firstContigClass, job_idx);

	debug3("created %d aligned classes for job %d",classCtr - firstContigClass, job_idx);
        lastAlignedClass = classCtr;
	i = 0;
	debug3("trying to create bids for job_idx %d firstContigClass %d lastAlignedClass %d",
	        job_idx, firstContigClass, lastAlignedClass);
	numClass = lastAlignedClass - firstContigClass;
	// the job has numClass classes 
	for (j = firstContigClass; j < lastAlignedClass; j++) {
	        if (bidCtr - job_array[job_idx].firstBid >= MAXBIDPERJOB)
	                break;
		ret = newBid(cArray[j].numCores, job_idx, j,
			job_array[job_idx].priority, cArray[j].preference, 
			cArray[j].numNodes, cArray[j].nodesUsed, job_array);
                if (ret <= 0)
                        break;
	}

	debug("going on to non-aligned contig bids");
	
	// contiguous bids
	for (i = 0; i < intervalCtr; i++) {
		if (classCtr >= MAXCLASSES) {
			debug3("max number of classes (%d) reached",MAXCLASSES);
			break;
		}		       
		if (classCtr - firstClass >= MAXCLASSESPERJOB) {
			debug3("max number of classes per job (%d) reached for job %d",MAXCLASSESPERJOB,job_array[job_idx].job_id);
			break;
		}
                if ( (intervals[i].gpu == job_array[job_idx].gpu) &&
                	(intervals[i].size >= job_array[job_idx].min_cpus) &&
//                	(intervals[i].size <= job_array[job_idx].max_cpus) &&
                	(intervals[i].numnodes >= job_array[job_idx].min_nodes) ) {
//                	(intervals[i].numnodes <= job_array[job_idx].max_nodes) ) {
                        int offset = job_array[job_idx].gpu * (nodeSize + 1);
                      	for (k = intervals[i].begin + 1; k < intervals[i].end - job_array[job_idx].min_nodes + 1; k++) {
                      	        //k = begin_node, starting from interval's initial point
                      	        //end_node, min(k + max_node,intervals[i].end);
                		if (classCtr >= MAXCLASSES) {
                			debug3("max number of classes (%d) reached",MAXCLASSES);
                			break;
                		}		       
                		if (classCtr - firstClass >= MAXCLASSESPERJOB) {
                			debug3("max number of classes per job (%d) reached for job %d",MAXCLASSESPERJOB,job_array[job_idx].job_id);
                			break;
                		}
                		end_node = MIN(job_array[job_idx].max_nodes, intervals[i].end - k + 1);
                        	cores = cumsum[k + end_node + offset] - cumsum[k + offset];
                        	debug3("begin node %d end_node %d cores %d ",k,end_node,cores);
                                if ( (cores >= job_array[job_idx].min_cpus) ) {
                                        ok = true;
                                        if (job_array[job_idx].cpus_per_node > 0)
                                        for (j = k; j < k + end_node; j++) {
                                                if (node_array[j].rem_cpus < job_array[job_idx].cpus_per_node) {
                                                        ok = false;
                                                }
                                        }
                                        if (ok) {
                                	cArray[classCtr].numNodes = end_node;
                                        if (job_array[job_idx].cpus_per_node > 0)
                                        	cArray[classCtr].numCores = end_node * job_array[job_idx].cpus_per_node;
                                        else
                                                cArray[classCtr].numCores = MIN(cores, job_array[job_idx].max_cpus);
                                        cArray[classCtr].numGpus = job_array[job_idx].gpu;
                                        cArray[classCtr].nodesUsed = (int *)malloc(cArray[classCtr].numNodes * sizeof(int));
                                        int sum = 0;
                                        for (j = k; j < k + end_node; j++) {
                                                sum += nArray[j];
                                        }
                                        cArray[classCtr].sumBids = sum;
                                        cArray[classCtr].preference = preferenceCalculation
                                                (cArray[classCtr].numNodes, nodeSize, 1,
                                                nodesetCount[cArray[classCtr].numGpus], 0.5, 0.0, 0.0);
                                        debug3("creating class for cores %d pref %.2f",cores,cArray[classCtr].preference);
                                        for (m = 0; m < cArray[classCtr].numNodes; m++) {
                                        	cArray[classCtr].nodesUsed[m] = k + m;
                                        	//debug3("nodeUsed %d: %d",m,cArray[classCtr].nodesUsed[m]);
                                        }
                                       	classCtr++;
                                       	}
				}
			}
		}
	}
	debug3("created %d contig classes for job %d",classCtr - lastAlignedClass, job_idx);

	debug3("created %d classes for job %d",classCtr - lastAlignedClass, job_idx);
        lastContClass = classCtr;
	i = bidCtr - job_array[job_idx].firstBid;
	debug3("trying to create bids for job_idx %d lastAlignedClass %d lastContClass %d",
	        job_idx, lastAlignedClass, lastContClass);
	numClass = lastContClass - lastAlignedClass;
	// the job has numClass classes 
	int *contigbids = (int *) malloc((MAXBIDPERJOB - i) * sizeof(int));
	k = 0;
	while ((i < MAXBIDPERJOB) && (i < numClass)) {
	        int max_pref_idx = -1;
	        double max_pref = 0.0;
	        int max_sum = LARGENUMBER;
	        for (j = lastAlignedClass; j < lastContClass; j++) {
	                for (l = 0; l < k; l++)
	                        if (contigbids[l] == j)
	                                continue;
                        if ((cArray[j].preference >= max_pref) && (cArray[j].sumBids <= max_sum)) {
                                max_pref = cArray[j].preference;
                                max_sum = cArray[j].sumBids;
                                max_pref_idx = j;
                        }	                
                }
                if (max_pref_idx == -1) {
                        break;
                }
		debug3("class with max pref (%.2f) is %d",max_pref, max_pref_idx);
		ret = newBid(cArray[max_pref_idx].numCores, job_idx, max_pref_idx,
			job_array[job_idx].priority, cArray[max_pref_idx].preference, 
			cArray[max_pref_idx].numNodes, cArray[max_pref_idx].nodesUsed, job_array);
                if (ret >= 0) {
                        i++;
                        contigbids[k++] = max_pref_idx;
                }
                else if (ret == -1)
                        break;
	}
	free(contigbids);


	debug("going on to non-contig bids");

	// non-contiguous bids
	// k: non-contiguousness level
	// depth first search
	
	if (job_array[job_idx].contiguous != 1) {

	for (i = 0; i < intervalCtr; i++) {
		temp.size = intervals[i].size;
		temp.gpu = intervals[i].gpu;
		temp.numnodes = intervals[i].numnodes;
		temp.begin = intervals[i].begin;
		temp.end = intervals[i].end;
		for (p = 0; p < temp.numnodes; p++) {
		        if (p > nodeSize)
        		        fatal("P %d IS GREATER THAN NODESIZE %d",p,nodeSize);
                        nodes[p] = intervals[i].begin + p;
                }
                debug3("testing non-contig nodeset starting from interval no %d",i);
		int contig = 1;
		for (j = i + 1; j < intervalCtr; j++) {
		        if (contig >= NONCONTIGLEVEL) {
		                debug3("max non contiguity level reached.");
		                break;
                        }
			if (classCtr >= MAXCLASSES) {
				debug3("max number of classes (%d) reached",MAXCLASSES);
				break;
			}		       
			if (classCtr - firstClass >= MAXCLASSESPERJOB) {
				debug3("max number of classes per job (%d) reached for job %d",MAXCLASSESPERJOB,job_array[job_idx].job_id);
				break;
			}

			if (intervals[j].gpu > intervals[i].gpu)
                                break;
			if (intervals[j].gpu == intervals[i].gpu) {
			        debug3("included interval no %d to the nodeset beginning with interval %d",j,i);
			        contig++;
			        //debug3("interval %d (%d nodes) and interval %d (%d nodes)have same gpu, combining",
			                 //j,intervals[j].numnodes,i,intervals[i].numnodes);
				temp.size += intervals[j].size;
				//debug3("size is now: %d", temp.size);
				//debug3("realloc'ed nodes");

				for (p = 0; p < intervals[j].numnodes; p++) {
				        //debug3("intervals[%d].nodes(%d) is: %d",j,p,intervals[j].nodes[p]);
                                        nodes[p + temp.numnodes] = intervals[j].begin + p;
                		        if (p + temp.numnodes > nodeSize)
                		                fatal("P %d +temp.numnodes %d IS GREATER THAN NODESIZE %d",p,temp.numnodes,nodeSize);
                                        //intervals[j].nodes[p];
				        //debug3("nodes element %d is %d",p + temp.numnodes,nodes[p + temp.numnodes]);	
				        //debug3("interval %d nodes %d: %d",j,p,intervals[j].nodes[p]);
                                }
				temp.numnodes += intervals[j].numnodes;
				//debug3("nodes in temp interval: (%d)",temp.numnodes);
				//xfor (p = 0; p < temp.numnodes; p++)
				        //xnodes[p] = 1;

		                if ( (temp.gpu == job_array[job_idx].gpu) &&
	        	        	(temp.size >= job_array[job_idx].min_cpus) &&
	        		       	(temp.numnodes >= job_array[job_idx].min_nodes) ) {
		                      	for (k = 0; k < temp.numnodes - job_array[job_idx].min_nodes + 1; k++) {
		                      	        if (classCtr >= MAXCLASSES) {
		                      	                debug3("max number of classes (%d) reached",MAXCLASSES);
		                      	                break;
                                                }
					       	if (classCtr - firstClass >= MAXCLASSESPERJOB) {
  				                        debug3("max number of classes per job (%d) reached for job %d",MAXCLASSESPERJOB,job_array[job_idx].job_id);
                        				break;
						}
						cores = 0;
						//debug3("job id: %d cores for noncontig class creation",job_array[job_idx].job_id);	
                                		end_node = MIN(job_array[job_idx].max_nodes, temp.numnodes - k);
						for (p = k; p < k + end_node; p++) {
							cores += node_array[nodes[p]].rem_cpus;
							//debug3("id: %d node %d remcpu %d total %d",
							//p,nodes[p],node_array[nodes[p]].rem_cpus,cores);
						}
						//debug("%d segments, %d cores", contig, cores);
		                                if ( (cores >= job_array[job_idx].min_cpus) ) {
		                                        ok = true;
                                                        if (job_array[job_idx].cpus_per_node > 0)
                                                                for (m = k; m < k + end_node; m++) {
                                                                        if (node_array[nodes[m]].rem_cpus < job_array[job_idx].cpus_per_node) {
                                                                                ok = false;
                                                                        }
                                                                }
                                                        if (ok) {
                                                        debug3("creating class for cores %d",cores);
                		                	cArray[classCtr].numNodes = end_node;
                                                        if (job_array[job_idx].cpus_per_node > 0)
                                                        	cArray[classCtr].numCores = end_node * job_array[job_idx].cpus_per_node;
                                                        else
                                                                cArray[classCtr].numCores = cores;
		                                        cArray[classCtr].numGpus = job_array[job_idx].gpu;
                		                        cArray[classCtr].nodesUsed = (int *)malloc(cArray[classCtr].numNodes * sizeof(int));
                                                        int sum = 0;
                                                        for (m = k; m < k + cArray[classCtr].numNodes; m++) {
                                                                sum += nArray[nodes[m]];
                                                        }
                                                        cArray[classCtr].sumBids = sum;
                                                        cArray[classCtr].preference = preferenceCalculation
                                                                (cArray[classCtr].numNodes, nodeSize, contig,
                                                                nodesetCount[cArray[classCtr].numGpus], 0.5, 0.0, 0.5);
		                                        for (m = 0; m < cArray[classCtr].numNodes; m++) {
		                                                //debug3("trying to access nodes %d : %d",k+m,nodes[k + m]);
                		                        	cArray[classCtr].nodesUsed[m] = nodes[k + m];
                                		       	}
                                		       	classCtr++;
                                		       	}
						}
					}
				}
			}
		}
                //free(nodes);
	}
	
	debug3("created %d classes for job %d",classCtr - lastContClass, job_idx);
        lastClass = classCtr;
	i = bidCtr - job_array[job_idx].firstBid;
	debug3("trying to create bids for job_idx %d firstContigClass %d lastClass %d",
	        job_idx, firstContigClass, lastClass);
	numClass = lastClass - lastContClass;
	// the job has numClass classes 
	k = 0;
        int *noncontigbids = (int *) malloc((MAXBIDPERJOB - i) * sizeof(int));
	while ((i < MAXBIDPERJOB) && (i < numClass)) {
	        int max_pref_idx = -1, l, ret;
	        double max_pref = 0.0;
	        int max_sum = LARGENUMBER;
//SIL
//	        for (j = lastContClass; j < lastClass; j++) {
//	                if (cArray[j].preference > preference) {
//	                        for (l = 0; l < k; l++) {
//	                                if (j == noncontigbids[l])
//	                                        continue;
//	                        }
//	                        max_pref_idx = j;
	                        //preference = cArray[j].preference;
	                //}
//                }
/////SILKAPA
	        for (j = lastContClass; j < lastClass; j++) {
	                for (l = 0; l < k; l++)
	                        if (noncontigbids[l] == j)
	                                continue;
                        if ((cArray[j].preference >= max_pref) && (cArray[j].sumBids <= max_sum)) {
                                max_pref = cArray[j].preference;
                                max_sum = cArray[j].sumBids;
                                max_pref_idx = j;
                        }	                
                }
                if (max_pref_idx == -1) {
                        break;
                }
		debug3("class with maximum preference (%.2f) is %d",max_pref, max_pref_idx);
		ret = newBid(cArray[max_pref_idx].numCores, job_idx, max_pref_idx,
			job_array[job_idx].priority, cArray[max_pref_idx].preference, 
			cArray[max_pref_idx].numNodes, cArray[max_pref_idx].nodesUsed, job_array);
                if (ret >= 0)
                        i++;
                else if (ret == -1)
                        break;
                noncontigbids[k++] = max_pref_idx;
	}
	free(noncontigbids);
	}

	debug("freeing nodes");
	job_array[job_idx].lastBid = bidCtr;
	free(nodes);
        debug("freed nodes");
}

// class generation for core-only jobs
void classGenerationC(solver_job_list_t *job_array, int job_idx, int nodeSize, sched_nodeinfo_t *node_array)
{ 
        int i, j = 0, k, l, cores, m, p, ret;
        int firstContigClass, firstClass, lastClass, lastContClass, lastAlignedClass, numClass;
        interval temp;
        int *nodes = (int*) malloc(nodeSize * sizeof(int));
        job_array[job_idx].firstBid = bidCtr;
        firstClass = classCtr;
        
        // slurmBids
/*
        bitstr_t *bitmap = bit_alloc (nodeSize);
        for (i = 0; i < nodeSize; i++) {
                if (temp_node_array[i].rem_cpus > 0) {
                        debug3("temp node array node %d remcpu %d",i,temp_node_array[i].rem_cpus);
                        bit_set (bitmap, (bitoff_t) (i));
                }
        }
        nodelist = bitmap2node_name(bitmap);
        debug3("in slurmBids; input to job_test for job %d (%d) input %s",
                job_array[job_idx].job_id, job_idx, nodelist);

        ret = select_g_job_test(job_array[job_idx].job_ptr, bitmap,
                job_array[job_idx].min_nodes, job_array[job_idx].max_nodes,
                job_array[job_idx].min_nodes, SELECT_MODE_WILL_RUN,
                NULL, NULL);
        if (ret == 0) {
                nodelist = bitmap2node_name(bitmap);
                debug3("in slurmBids; job_test for job %d (%d)returned %s",
                        job_array[job_idx].job_id, job_idx, nodelist);
                // buraya bid olusturmak gerekli, bitmap'teki nodelara gore.
                int sum = 0;
                for (i = 0; i < nodeSize; i++) {
                        if (bit_test(bitmap, i)) {
                                sum += temp_node_array[i].rem_cpus;
                        }
                }
                if (sum >= job_array[job_idx].min_cpus) {
                int numNodes = bit_set_count(bitmap);
                int *bidNodes = (int*)malloc(numNodes * sizeof(int));
                int j = 0;
                int contig = 0;
                for (i = 0; i < nodeSize; i++) {
                        if (bit_test(bitmap, i)) {
                                bidNodes[j] = i;
                                j++;
                        }
                }
                
                for (i = 1; i < j; i++) {
                        if (bidNodes[i] - bidNodes[i-1] > 1)
                                contig++;
                }
                debug3("in slurmBids; job_test for job %d is on %d contig parts", job_array[job_idx].job_id, contig);
                
                cArray[classCtr].numNodes = numNodes;
                cArray[classCtr].nodesUsed = (int *)malloc(cArray[classCtr].numNodes * sizeof(int));
                int classIdx = classCtr++;
                
                double priority = job_array[job_idx].priority;
                int amount = MIN(job_array[job_idx].max_cpus, sum);
                double preference = preferenceCalculation
                        (numNodes, nodeSize, contig,
                        nodesetCount[job_array[job_idx].gpu], 0.5, 0.0, 0.5);
                newBid(amount, job_idx, classIdx, priority, preference, numNodes, bidNodes, job_array);
                // sonra da temp_node_array update
                for (j = 0; j < numNodes; j++) {
                        k = bidNodes[j];
                        if (job_array[job_idx].cpus_per_node > 0) {
                                temp_node_array[k].rem_cpus -= job_array[job_idx].cpus_per_node;
                        } else {
                                temp_node_array[k].rem_cpus = 0;
                        }
                }
		}
        } else {
                debug3("in slurmBids; job_test for job %d returned false value %d", job_idx, ret);
        }
*/
        
        debug3("entering interval loop, intervalCtr %d, job id %d",intervalCtr,job_array[job_idx].job_id);
	firstContigClass = classCtr;
        for (i = 0; i < intervalCtr; i++) {
                 if (intervals[i].gpu != job_array[job_idx].gpu)
                        continue;
                 if ( (intervals[i].size < job_array[job_idx].min_cpus) )
			break;
	                 for (k = intervals[i].end; k >= intervals[i].begin; k--) {
                                      if (classCtr >= MAXCLASSES)
                                              break;
                                      if (classCtr - firstClass >= MAXCLASSESPERJOB)
                                              break;
                                      cores = cumsum[intervals[i].end] - cumsum[k - 1];
                                      if ( (cores < job_array[job_idx].min_cpus) || (intervals[i].end + 1 - k > job_array[job_idx].min_cpus)) 
				   	      continue;
                                      
					debug3("generating right-aligned bid for job %d",job_array[job_idx].job_id);
                                        cArray[classCtr].numNodes = intervals[i].end - k + 1;
                                        cArray[classCtr].numCores = cores;
                                        cArray[classCtr].numGpus = job_array[job_idx].gpu;
                                        debug4("creating nodesUsed size %d",cArray[classCtr].numNodes);
                                        cArray[classCtr].nodesUsed = (int *)malloc(cArray[classCtr].numNodes * sizeof(int));
                                        cArray[classCtr].preference = preferenceCalculation
                                                      (cArray[classCtr].numNodes, nodeSize, 1,
                                                      nodesetCount[0], 0.0, 0.25, 0.0);
                                        debug4("filling nodesUsed");
                                        for (m = 0; m < cArray[classCtr].numNodes; m++) {
                                                      debug5("nodesUsed element %d is %d",m,k+m);
                                                      cArray[classCtr].nodesUsed[m] = k + m;
                                        }  
                                        classCtr++;
                                        //debug3("class ctr now %d",classCtr);
                                        break;
                             }
                             for (k = intervals[i].begin; k <= intervals[i].end; k++) {
                                      if (classCtr >= MAXCLASSES)
                                              break;
                                      if (classCtr - firstClass >= MAXCLASSESPERJOB)
                                              break;
                                      cores = cumsum[k + 1] - cumsum[intervals[i].begin];
                                      if ( (cores < job_array[job_idx].min_cpus) || (k + 1 - intervals[i].begin > job_array[job_idx].min_cpus))
                                              continue;
                                      
                                              debug3("generating left-aligned bid for job %d",job_array[job_idx].job_id);
                                              cArray[classCtr].numNodes = k - intervals[i].begin + 1;
                                              cArray[classCtr].numCores = cores;
                                              cArray[classCtr].numGpus = job_array[job_idx].gpu;
                                              debug4("creating nodesUsed size %d",cArray[classCtr].numNodes);
                                              cArray[classCtr].nodesUsed = (int *)malloc(cArray[classCtr].numNodes * sizeof(int));
                                              cArray[classCtr].preference = preferenceCalculation
                                                      (cArray[classCtr].numNodes, nodeSize, 1,
                                                      nodesetCount[0], 0.0, 0.25, 0.0);
                                              debug4("filling nodesUsed");
                                              for (m = 0; m < cArray[classCtr].numNodes; m++) {
                                                      debug5("nodesUsed element %d is %d",m,k+m);
                                                      cArray[classCtr].nodesUsed[m] = intervals[i].begin + m;
                                              }  
                                              classCtr++;
                                              //debug3("class ctr now %d",classCtr);
                                              break;
                             }
        }

	debug3("created %d aligned classes for job %d",classCtr - firstContigClass, job_idx);
        lastAlignedClass = classCtr;
	i = 0;
	debug3("trying to create aligned bids for job_idx %d firstContigClass %d lastAlignedClass %d",
	        job_idx, firstContigClass, lastAlignedClass);
	numClass = lastAlignedClass - firstContigClass;
	// the job has numClass classes 
	for (j = firstContigClass; j < lastAlignedClass; j++) {
	        if (bidCtr - job_array[job_idx].firstBid >= MAXBIDPERJOB)
	                break;
		ret = newBid(job_array[job_idx].min_cpus, job_idx, j,
			job_array[job_idx].priority, cArray[j].preference, 
			cArray[j].numNodes, cArray[j].nodesUsed, job_array);
        	if (ret == -1)
        	        break;
                else if (ret >= 0)
                        i++;
	}

	// non-contiguous bids 
	// k: non-contiguousness level
	// depth first search 

        for (i = 0; i < intervalCtr; i++) {
                 if (intervals[i].gpu != job_array[job_idx].gpu)
                        continue;
                 if ( (intervals[i].size >= job_array[job_idx].min_cpus) ) {
			for (l = intervals[i].begin; l <= intervals[i].end; l++) {
                             for (k = intervals[i].begin; k <= intervals[i].end; k++) {
                                      if (classCtr >= MAXCLASSES)
                                              break;
                                      if (classCtr - firstClass >= MAXCLASSESPERJOB)
                                              break;
                                      cores = cumsum[k + 1] - cumsum[l];
					if ( (cores < job_array[job_idx].min_cpus) || (k + 1 - l > job_array[job_idx].min_cpus) )
						continue;
                                      int sum = 0;
                                      for (j = l; j <= k; j++) {
                                              sum += nArray[j];
                                      }
                                              cArray[classCtr].numNodes = k + 1 - l;
                                              cArray[classCtr].numCores = cores;
                                              cArray[classCtr].numGpus = job_array[job_idx].gpu;
                                              debug4("creating nodesUsed size %d",cArray[classCtr].numNodes);
                                              cArray[classCtr].nodesUsed = (int *)malloc(cArray[classCtr].numNodes * sizeof(int));
                                              cArray[classCtr].preference = preferenceCalculation
                                                      (cArray[classCtr].numNodes, nodeSize, 1,
                                                      nodesetCount[0], 0.25, 0.25, 0.0);
                                              debug4("filling nodesUsed");
                                              for (m = 0; m < cArray[classCtr].numNodes; m++) {
                                                      debug5("nodesUsed element %d is %d",m,k+m);
                                                      cArray[classCtr].nodesUsed[m] = l + m;
                                              }  
                                        cArray[classCtr].sumBids = sum;
                                              classCtr++;
                                              //debug3("class ctr now %d",classCtr);
                                              j++;
						break;
                             }
			}
                 }
        }

	debug3("created %d classes for job %d",classCtr - lastAlignedClass, job_idx);
        lastContClass = classCtr;
	i = 0;
	debug3("trying to create bids for job_idx %d lastAlignedClass %d lastContClass %d",
	        job_idx, lastAlignedClass, lastContClass);
	numClass = lastContClass - lastAlignedClass;
	// the job has numClass classes
	int *contigbids = (int *) malloc((MAXBIDPERJOB - i) * sizeof(int));
	k = 0;

	while ((i < MAXBIDPERJOB) && (i < numClass)) {
	        int max_pref_idx = -1;
	        double max_pref = 0.0;
		int max_sum = LARGENUMBER;
	        for (j = lastAlignedClass; j < lastContClass; j++) {
	                for (l = 0; l < k; l++)
	                        if (contigbids[l] == j)
	                                continue;
                        if ((cArray[j].preference >= max_pref) && (cArray[j].sumBids <= max_sum)) {
                                max_pref = cArray[j].preference;
				max_sum = cArray[j].sumBids;
                                max_pref_idx = j;
                        }	                
                }
                if (max_pref_idx == -1) {
                        break;
                }
		debug3("class with max pref (%.2f) is %d",max_pref, max_pref_idx);
		ret = newBid(cArray[max_pref_idx].numCores, job_idx, max_pref_idx,
			job_array[job_idx].priority, cArray[max_pref_idx].preference, 
			cArray[max_pref_idx].numNodes, cArray[max_pref_idx].nodesUsed, job_array);
                if (ret >= 0) {
                        i++;
                        contigbids[k++] = max_pref_idx;
                }
                else if (ret == -1)
                        break;
	}
	free(contigbids);

	// non-contiguous bids 
	// k: non-contiguousness level 
	// depth first search 

	if (job_array[job_idx].contiguous != 1) {
	for (i = 0; i < intervalCtr; i++) {
	        if (intervals[i].gpu != job_array[job_idx].gpu)
	                continue;
		temp.size = intervals[i].size;
		temp.numnodes = intervals[i].numnodes;
		temp.begin = intervals[i].begin;
		temp.end = intervals[i].end;
		for (p = 0; p < temp.numnodes; p++)
                        nodes[p] = intervals[i].begin + p;
		int contig = 1;
		for (j = i + 1; j < intervalCtr; j++) {
			if (classCtr >= MAXCLASSES) {
				debug3("max number of classes (%d) reached",MAXCLASSES);
				break;
			}		       
			if (classCtr - firstClass >= MAXCLASSESPERJOB) {
				debug3("max number of classes per job (%d) reached for job %d",MAXCLASSESPERJOB,job_array[job_idx].job_id);
				break;
			}

			if (intervals[j].gpu != job_array[job_idx].gpu)
                                continue;
			        contig++;
			        if (contig >= NONCONTIGLEVEL) {
			                debug3("max noncontig level (%d) reached",NONCONTIGLEVEL);
			                break;
			        }
				temp.size += intervals[j].size;

				for (p = 0; p < intervals[j].numnodes; p++) {
				        //debug3("intervals[%d].nodes(%d) is: %d",j,p,intervals[j].nodes[p]);
                                        nodes[p + temp.numnodes] = intervals[j].begin + p;
                                        //intervals[j].nodes[p];
				        //debug3("nodes element %d is %d",p + temp.numnodes,nodes[p + temp.numnodes]);	
				        //debug3("interval %d nodes %d: %d",j,p,intervals[j].nodes[p]);
                                }
				temp.numnodes += intervals[j].numnodes;
				//xfor (p = 0; p < temp.numnodes; p++)
				        //xnodes[p] = 1;

		                if ( (temp.size >= job_array[job_idx].min_cpus) ) {
		                      	for (k = 0; k < temp.numnodes; k++) {
		                      	        if (classCtr >= MAXCLASSES) {
		                      	                debug3("max number of classes (%d) reached",MAXCLASSES);
		                      	                break;
                                                }
					       	if (classCtr - firstClass >= MAXCLASSESPERJOB) {
  				                        debug3("max number of classes per job (%d) reached for job %d",MAXCLASSESPERJOB,job_array[job_idx].job_id);
                        				break;
						}
						cores = 0;	
						for (p = 0; p < k; p++) {
						        //debug("nodes[p:%d]: %d",p,nodes[p]);
							cores += node_array[nodes[p]].rem_cpus;
						}
						//debug("int i: %d, j: %d, %d segments, %d cores", i, j, contig, cores);
		                                if (cores < job_array[job_idx].min_cpus)
							continue;

                		                	cArray[classCtr].numNodes = k;
                                			cArray[classCtr].numCores = cores;
		                                        cArray[classCtr].numGpus = job_array[job_idx].gpu;
                		                        cArray[classCtr].nodesUsed = (int *)malloc(cArray[classCtr].numNodes * sizeof(int));
                                                        cArray[classCtr].preference = preferenceCalculation
                                                                (cArray[classCtr].numNodes, nodeSize, contig,
                                                                nodesetCount[0], 0.5, 0.25, 0.25);		                                        
                                                        for (m = 0; m < cArray[classCtr].numNodes; m++)
                		                        	cArray[classCtr].nodesUsed[m] = nodes[m];
                                		       	classCtr++;
                                                        int sum = 0;
                                                        for (m = 0; m < k; m++) {
                                                                sum += nArray[nodes[m]];
                                                        }
		                                        cArray[classCtr].sumBids = sum;
							break;
					}
				}
		}
                //free(nodes);
	}
        lastClass = classCtr;
	numClass = lastClass - lastContClass;
	
	i = 0;
        debug3("searched suitable intervals, lastContClass: %d, lastClass: %d", lastContClass, lastClass);
	k = 0;
        int *noncontigbids = (int *) malloc((MAXBIDPERJOB - i) * sizeof(int));
	while ((i < MAXBIDPERJOB) && (i < numClass)) {
	        int max_pref_idx = -1, l, ret;
	        double max_pref = 0.0;
		int max_sum = LARGENUMBER;
                for (j = lastContClass; j < lastClass; j++) {
                        for (l = 0; l < k; l++)
                                if (noncontigbids[l] == j)
                                        continue;
                        if ((cArray[j].preference >= max_pref) && (cArray[j].sumBids <= max_sum)) {
                                max_pref = cArray[j].preference;
                                max_sum = cArray[j].sumBids;
                                max_pref_idx = j;
                        }
                }
                if (max_pref_idx == -1) {
                        break;
                }

                if (max_pref_idx == -1) {
                        break;
                }
		debug3("class with maximum preference (%.2f) is %d", max_pref, max_pref_idx);
		ret = newBid(cArray[max_pref_idx].numCores, job_idx, max_pref_idx,
			job_array[job_idx].priority, cArray[max_pref_idx].preference, 
			cArray[max_pref_idx].numNodes, cArray[max_pref_idx].nodesUsed, job_array);
                if (ret >= 0)
                        i++;
                else if (ret == -1)
                        break;
                noncontigbids[k++] = max_pref_idx;
	}
	free(noncontigbids);
	}

	free(nodes);
	job_array[job_idx].lastBid = bidCtr;
}

                        
                        

inline int
populatebynonzero (CPXENVptr env, CPXLPptr lp, int m, int n,
		sched_nodeinfo_t *node_array, solver_job_list_t* job_array)
{
        //jobNodeCtr--;
	int NUMCOLS = bidCtr + 2 * jobNodeCtr; 
	int NUMROWS = 2 * n + bidCtr + 2 * m + 1 * jobNodeCtr + n * m;
	int NUMNZ = 3 * bidCtr + 11 * (jobNodeCtr);
	int NZc = 0; /* nonzero counter */
	
	int status = 0;
	double *obj = NULL;
	obj = (double*)malloc(NUMCOLS * sizeof(double));
	double *lb = (double*)malloc(NUMCOLS * sizeof(double));
	double *ub = (double*)malloc(NUMCOLS * sizeof(double));
	double *rhs = (double*)malloc(NUMROWS * sizeof(double));
	char *ctype = (char*)malloc(NUMCOLS * sizeof(char));
	char *sense = (char*)malloc(NUMROWS * sizeof(char));
	/*
	char **colname = (char**)malloc(NUMCOLS * sizeof(char*));
	char str[10];
	*/
	int *rowlist = (int*)malloc(NUMNZ * sizeof(int));
	int *collist = (int*)malloc(NUMNZ * sizeof(int));
	double *vallist = (double*)malloc(NUMNZ * sizeof(double));
	int i, j, k, c, tempCtr;
	
	CPXchgobjsen (env, lp, CPX_MAX);  /* Problem is maximization */
	
	debug3("bidCtr %d jobNodeCtr %d NUMROWS %d NUMCOLS %d", bidCtr, jobNodeCtr,NUMROWS,NUMCOLS);
	debug3("n %d m %d NUMNZ %d", n, m, NUMNZ);

	/* row definitions */
	
	/* constraints (1) b_jc <= 1 */
	for (j = 0; j < n; j++) {
		sense[j] = 'L';
		rhs[j] = 1.0;
	}
	
	/* constraints (2) and (3); = 0 */
	for (j = n; j < n * 2 + bidCtr; j++) {
		sense[j] = 'E';
		rhs[j] = 0.0;
	}
	
	/* constraints (4) and (5) */
	for (j = n * 2 + bidCtr; j < n * 2 + bidCtr + m; j++) {
	        i = j - (n * 2 + bidCtr);
		sense[j] = 'L';
		rhs[j] = 1.0 * node_array[i].rem_cpus;
		sense[m + j] = 'L';
		rhs[m + j] = 1.0 * node_array[i].rem_gpus;
	}
	
	/* constraint (6)  */
	for (j = n * 2 + bidCtr + 2 * m ; j < n * 2 + bidCtr + 2 * m + jobNodeCtr; j++) {
		sense[j] = 'L';
		rhs[j] = 0;
	}
	
	/* constraint (8) */
	c = 2 * n + bidCtr + 2 * m + jobNodeCtr;
	for (j = c; j < c + m * n; j++) {
	        sense[j] = 'E';
	        rhs[j] = 0;
	}
	
	status = CPXnewrows (env, lp, NUMROWS, rhs, sense, NULL, NULL);
	if ( status ) goto TERMINATE;

	/* column definitions */	

	/* b_jc definitions */
	for (j = 0; j < bidCtr; j++) {
		/*
		sprintf(str, "b_%d",j);
		colname[j] = str;
		*/
		ctype[j] = CPX_BINARY;
		lb[j] = 0.0;
		ub[j] = 1.0;
		
	}
	
	/* u_jn definitions */
	for (j = 0; j < jobNodeCtr; j++) {
		/*
		sprintf(str, "u_%d",j);
		colname[bidCtr + j] = str;
		*/
		ctype[bidCtr + j] = CPX_BINARY;
		lb[bidCtr + j] = 0.0;
		ub[bidCtr + j] = 1.0;
	}	
		
	/* r_jn definitions */
	for (j = 0; j < jobNodeCtr; j++) {
		/*
		sprintf(str, "r_%d",j);
		colname[bidCtr + jobNodeCtr + j] = str;
		*/
		ctype[bidCtr + jobNodeCtr + j] = CPX_INTEGER;
		lb[bidCtr + jobNodeCtr + j] = 0.0;
		ub[bidCtr + jobNodeCtr + j] = CPX_INFBOUND;
	}
		
	for (k = 0; k < bidCtr; k++) {
		debug3("bid %d prio %f preference %f",k,bids[k].priority,bids[k].preference);
		obj[k] = (bids[k].priority + bids[k].preference * alpha);
	}
	
	for (k = bidCtr; k < NUMCOLS; k++) {
	        obj[k] = 0.0;
	}
	
	status = CPXnewcols (env, lp, NUMCOLS, obj, lb, ub, ctype, NULL);
	if ( status )  goto TERMINATE;

	/* constraints */
	
	/* constraint (1) coefficients: 
	   sum_bjc <= 1, c in B_j, for all j */
	for (j = 0; j < n; j++) {
	        debug("job %d firstbid %d lastbid %d",j,job_array[j].firstBid,job_array[j].lastBid);
		for (k = job_array[j].firstBid; k < job_array[j].lastBid; k++) {
			rowlist[NZc] = j;
			debug("row %d, col %d, NZc %d",j,k,NZc);
			collist[NZc] = k;
                        debug3("col index of %d is %d",NZc, collist[NZc]);
			vallist[NZc++] = 1.0;
		}
	}
	/* NZc = bidCtr */
	debug3("NZc is now %d, should be %d",NZc, bidCtr);

	debug("constraint2 start");
	/* constraint (2) coefficients: sum_ujn = b_jc * R_J^node */
	/* bundan emin ol, yanlis gibi sanki */
	tempCtr = n;
	for (j = 0; j < n; j++) {
                //debug("job %d first bid %d last bid %d", j, job_array[j].firstBid, job_array[j].lastBid);
		for (k = job_array[j].firstBid; k < job_array[j].lastBid; k++) {
			for (i = bids[k].firstNode; i < bids[k].lastNode; i++) {
					rowlist[NZc] = tempCtr;
					collist[NZc] = bidCtr + jobNodeIdx[i];
                                        debug3("col index of %d is %d",NZc, collist[NZc]);
					//if (job_array[j].min_nodes == 0)
                                        	//vallist[NZc++] = 0.0;
					//if (job_array[j].min_nodes > 0)
                                        	vallist[NZc++] = 1.0;
					//debug("sum_ujn row %d, col %d coef 1.0 (nodeName %d)",tempCtr, bidCtr + jobNodeIdx[i], jobNodeName[i]);
			}			
			rowlist[NZc] = tempCtr++;
			collist[NZc] = k;
                        debug3("col index of %d is %d",NZc, collist[NZc]);
			if (job_array[j].min_nodes > 0)
                        	vallist[NZc++] = (-1.0 * bids[k].numNodes);
			if (job_array[j].min_nodes == 0)
                        	vallist[NZc++] = (-1.0 * bids[k].numNodes);
			//debug("sum_ujn row %d col %d coef %d",tempCtr - 1, k, -job_array[j].nodes);
		}
	}
	/* NZc = bidCtr + bidCtr + jobNodeCtr */
	debug3("NZc is now %d, should be %d",NZc, bidCtr * 2 + jobNodeCtr);

	debug("constraint 3 start");
	/* constraint (3) coefficients: sum_ujn + r_jn = R_J^cpu * b_jc */
	tempCtr = n + bidCtr;
	for (j = 0; j < n; j++) {
		for (k = job_array[j].firstBid; k < job_array[j].lastBid; k++) {
			/* sum _bjc */
			rowlist[NZc] = tempCtr;
			collist[NZc] = k;
			vallist[NZc++] = -bids[k].amount;
			//debug("sum_bjc row %d col %d coef %d", tempCtr, k, -job_array[j].min_cpus);
		
			for (i = bids[k].firstNode; i < bids[k].lastNode; i++) {
				/* sum u_jn */
				rowlist[NZc] = tempCtr;
				collist[NZc] = bidCtr + jobNodeIdx[i];
                       	        debug3("col index of %d is %d",NZc, collist[NZc]);
				vallist[NZc++] = 1.0;
				//debug("sum_ujn row %d col %d val 1", tempCtr, bidCtr + jobNodeIdx[i]);
			
				/* sum r_jn */
				rowlist[NZc] = tempCtr;
				collist[NZc] = bidCtr + jobNodeCtr + jobNodeIdx[i];
                       	        debug3("col index of %d is %d",NZc, collist[NZc]);
				vallist[NZc++] = 1.0;
				//debug("sum_rjn row %d col %d val 1", tempCtr, bidCtr + jobNodeCtr + jobNodeIdx[i]);
			}
		}
		tempCtr++;
	}
	/* NZc = 2 * bidCtr + jobNodeCtr + 2 * jobNodeCtr + bidCtr */
	debug3("NZc is now %d, should be %d",NZc, 3 * bidCtr + 3 * jobNodeCtr);
	
        debug("constraint 4 start");
	/* constraint (4) coefficients: u_jn + r_jn <= A_n^cpu */
	tempCtr = 2 * n + bidCtr;
	for (j = 0; j < n; j++) {
		for (k = job_array[j].firstBid; k < job_array[j].lastBid; k++) {
			for (c = bids[k].firstNode; c < bids[k].lastNode; c++) {
				rowlist[NZc] = tempCtr + jobNodeName[c];
				collist[NZc] = bidCtr + jobNodeIdx[c];
                       	        debug3("col index of %d is %d",NZc, collist[NZc]);
                                vallist[NZc++] = 1.0;
				//debug("cpu sum row %d col %d val 1",n + 2 * bidCtr + i, bidCtr + jobNodeIdx[c]);
				
				rowlist[NZc] = tempCtr + jobNodeName[c];
				collist[NZc] = bidCtr + jobNodeCtr + jobNodeIdx[c];
                       	        debug3("col index of %d is %d",NZc, collist[NZc]);
				vallist[NZc++] = 1.0;
				//debug("cpu sum row %d col %d val 1",n + 2 * bidCtr + i, bidCtr + jobNodeCtr + jobNodeIdx[c]);
                        }
                }
        }
	/* NZc = 3 * bidCtr + 3 * jobNodeCtr + 2 * jobNodeCtr */
	debug3("NZc is now %d, should be %d",NZc, 3 * bidCtr + 5 * jobNodeCtr);

        debug("constraint5 start");
	/* constraint (5) coefficients: u_jn * g_j <= A_n^gpu */
	for (j = 0; j < n; j++) {
		for (k = job_array[j].firstBid; k < job_array[j].lastBid; k++) {
                        for (c = bids[k].firstNode; c < bids[k].lastNode; c++) {
				rowlist[NZc] = n *2 + bidCtr + m + jobNodeName[c];
				collist[NZc] = bidCtr + jobNodeIdx[c];
                       	        debug3("col index of %d is %d",NZc, collist[NZc]);
				vallist[NZc++] = 1.0 * job_array[j].gpu;
				//debug("cpu sum row %d col %d val %d",n + 2 * bidCtr + m + i, bidCtr + jobNodeIdx[c], job_array[j].gpu);
			}
                }
        }
	/* NZc = 3 * bidCtr + 6 * jobNodeCtr */
	debug3("NZc is now %d, should be %d",NZc, 3 * bidCtr + 6 * jobNodeCtr);
	
        debug("constraint6 start");
	/* constraint (6); 
	r_jn - u_jn * MIN(A_n^cpu - 1 , R_j^cpu - 1) <= 0
	*/
        tempCtr = n * 2 + bidCtr + m * 2;
        for (j = 0; j < n; j++) {
                for (k = job_array[j].firstBid; k < job_array[j].lastBid; k++) {
			for (c = bids[k].firstNode; c < bids[k].lastNode; c++) {
				rowlist[NZc] = tempCtr;
				collist[NZc] = bidCtr + jobNodeCtr + jobNodeIdx[c];
                       	        debug3("col index of %d is %d",NZc, collist[NZc]);
				vallist[NZc++] = 1.0;
				//debug("rjn<=ujn row %d col %d val 1",tempCtr, bidCtr + jobNodeCtr + jobNodeIdx[c]);
				
				rowlist[NZc] = tempCtr++;
				collist[NZc] = bidCtr + jobNodeIdx[c];
				i = jobNodeName[c];
                       	        debug3("col index of %d is %d",NZc, collist[NZc]);
				vallist[NZc++] = -1.0 * MIN(bids[k].amount - 1, node_array[i].rem_cpus - 1);
				//debug("rjn<=ujn sum row %d col %d val %.2f", tempCtr, bidCtr + jobNodeIdx[c],-1.0 * MIN(job_array[j].min_cpus - 1, node_array[i].rem_cpus - 1));
                        }
		}
	}
	/* NZc = 3 * bidCtr + 8 * jobNodeCtr */
	debug3("NZc is now %d, should be %d",NZc, 3 * bidCtr + 8 * jobNodeCtr);
	
        debug("constraint8 start");
	/* constraint (8);
        IF CPUS_PER_NODE IS SET: 
        u_jn + r_jn = \sum for all c \in B_j cpus_per_node * b_jc 
        */
        /*
	tempCtr = 2 * n + bidCtr + 2 * m + jobNodeCtr;
	for (j = 0; j < n; j++) {
		for (k = job_array[j].firstBid; k < job_array[j].lastBid; k++) {
			for (i = bids[k].firstNode; i < bids[k].lastNode; i++) {
        */
				/* u_jn */
				/*
				rowlist[NZc] = tempCtr;
				collist[NZc] = bidCtr + jobNodeIdx[i];
				if (job_array[j].cpus_per_node > 0)
        				vallist[NZc++] = 1.0;
                                else
                                        vallist[NZc++] = 0.0;
                                */
				/* r_jn */
				/*
				rowlist[NZc] = tempCtr;
				collist[NZc] = bidCtr + jobNodeCtr + jobNodeIdx[i];
				if (job_array[j].cpus_per_node > 0)
        				vallist[NZc++] = 1.0;
                                else
                                        vallist[NZc++] = 0.0;
                                
        			rowlist[NZc] = tempCtr;
        			collist[NZc] = k;
        			vallist[NZc++] = -1.0 * (double)(job_array[j].cpus_per_node);
        			
        		}
                        tempCtr++;
		}
	}
	*/
	
	tempCtr = 2 * n + bidCtr + 2 * m + jobNodeCtr;
	for (j = 0; j < n; j++) {
	        for (k = job_array[j].firstBid; k < job_array[j].lastBid; k++) {
	                for (i = bids[k].firstNode; i < bids[k].lastNode; i++) {
        	                /* u_jn */
        	                rowlist[NZc] = tempCtr + j * m + jobNodeName[i];
        	                debug3("r_jn u_jn in row %d, j %d i %d",tempCtr + j * m + i, j, i);
        	                collist[NZc] = bidCtr + jobNodeIdx[i];
                       	        debug3("col index of %d is %d",NZc, collist[NZc]);
        			if (job_array[j].cpus_per_node > 0)
               				vallist[NZc++] = 1.0;
                                else
                                        vallist[NZc++] = 0.0;
                                        
                                /* r_jn */
                                rowlist[NZc] = tempCtr + j * m + jobNodeName[i];
        			collist[NZc] = bidCtr + jobNodeCtr + jobNodeIdx[i];
                       	        debug3("col index of %d is %d",NZc, collist[NZc]);
        			if (job_array[j].cpus_per_node > 0)
                			vallist[NZc++] = 1.0;
                                else
                                        vallist[NZc++] = 0.0;
                        }
	        }
	}
	/* NZc = 3 * bidCtr + 8 * jobNodeCtr + m * n * 2 */
	debug3("NZc is now %d, should be %d",NZc, 3 * bidCtr + 10 * jobNodeCtr);
	debug3("constraint 8, u_jn r_jn done. NZc is %d",NZc);
	
	for (j = 0; j < n; j++) {
	        for (k = job_array[j].firstBid; k < job_array[j].lastBid; k++) {
	                for (i = bids[k].firstNode; i < bids[k].lastNode; i++) {
	                        /* b_jc */
	                        rowlist[NZc] = tempCtr + j * m + jobNodeName[i];
	                        debug3("j %d k %d i %d temp %d in row %d",
	                                j, k, i, tempCtr, rowlist[NZc]);
	                        collist[NZc] = k;
                       	        debug3("col index of %d is %d",NZc, collist[NZc]);
	                        vallist[NZc++] = -1.0 * (double)(job_array[j].cpus_per_node);
                        }
                }
	}
	/* NZc = 3 * bidCtr + 9 * jobNodeCtr + m * n * 2 */	
	debug3("NZc is now %d, should be %d",NZc, 3 * bidCtr + 11 * jobNodeCtr);
	
	debug3("constraint 8, b_jc done. NZc is %d",NZc);
	/*nzc = num_bids*/
	
	status = CPXchgcoeflist (env, lp, NZc, rowlist, collist, vallist);   
	debug3("status from change coef list: %d",status);
	if ( status )  goto TERMINATE;

TERMINATE:
	free_and_null ((char **) &obj);

	free_and_null ((char **) &lb);
	free_and_null ((char **) &ub);
	free_and_null ((char **) &rhs);
	free_and_null ((char **) &sense);
	free_and_null ((char **) &rowlist);
	free_and_null ((char **) &collist);
	free_and_null ((char **) &vallist);
	debug("freed all, bb");

	return (status);
} 



/* 
returns 0 if solution found
	2 if time limit hit
*/
extern int solve_allocation(int nodeSize, int windowSize, int timeout, 
			sched_nodeinfo_t *node_array, 
			solver_job_list_t *job_array, int max_bid_count)
{
	char filename[256];
	solver_job_list_t *sjob_ptr;
	struct job_details *job_det_ptr;
	int n = windowSize, m = nodeSize, k, c, i;
	bool half = false;
	uint32_t begin_time = time(NULL), mid_time;
	uint32_t time1;
	uint16_t dummy;
	double objval;
	double *x = NULL;
	double *pi = NULL;
	double *slack = NULL;
	double *dj = NULL;

	CPXENVptr env = NULL;
	CPXLPptr lp = NULL;
	int status = 0, status2 = 0, j, cur_numrows, cur_numcols;
	FILE *log;
	
	bidCtr = 0;
	classMapCtr = 0;
	classCtr = 0;
	jobNodeCtr = 0;
	maxBids = max_bid_count;
	
	jobNodeIdx = (int *)malloc(MAXJOBNODE * 10 * sizeof(int));
	jobNodeName = (int *)malloc(MAXJOBNODE * 10 * sizeof(int));
	// number of class types cannot exceed window size
	//classMap = (class_map *)malloc(windowSize * sizeof(class_map));
	intervals = (interval *)malloc(MAXINTERVALS * sizeof(interval));
	nArray = (int *)malloc(nodeSize * sizeof(int));
	for (i = 0; i < nodeSize; i++)
	        nArray[i] = 0;
	
	/* find minPrioDiff */
	double minPrio = 10000000.0;
	for (i = 0; i < windowSize; i++) {
                if (job_array[i].priority < minPrio) {
                        minPrio = job_array[i].priority;
                }
	}
	        
	
	/* call class generation */
	/*
	classListCreation(nodeSize, node_array);
	classGeneration(nodeSize, node_array);
	*/
	
	/* cumsum generation */
	time1 = time(NULL);
	cumSumGeneration(nodeSize, node_array);
	debug2("timer: cumsum took %u",(uint16_t)(time(NULL)-time1));
	
	/* call bid generation 
	   this includes class generation 
	*/
	bids = (bid *)malloc(maxBids * sizeof(bid));
	cArray = (class *)malloc(MAXCLASSES * sizeof(class));
	time1 = time(NULL);

	//slurmBids(job_array, nodeSize, windowSize, node_array);
        //slurmBids icin
        temp_node_array = (sched_nodeinfo_t*)
                malloc(node_record_count * sizeof(sched_nodeinfo_t));
        for (i = 0; i < nodeSize; i++) {
                temp_node_array[i].rem_cpus = node_array[i].rem_cpus;
        }


	for (j = 0; j < windowSize; j++) {
	        if (classCtr >= MAXCLASSES) {
	                debug3("max classes reached in job %d of window %d",j,windowSize);
 		        job_array[j].firstBid = bidCtr;
 		        job_array[j].lastBid = bidCtr;
                } else if (bidCtr >= maxBids) {
	                debug3("max bids reached in job %d of window %d",j,windowSize);
 		        job_array[j].firstBid = bidCtr;
 		        job_array[j].lastBid = bidCtr;
                } else if (job_array[j].min_nodes > 0) {
 		        classGenerationN(job_array, j, nodeSize, node_array);
 		} else {
                        classGenerationC(job_array, j, nodeSize, node_array);
                }
	}

        for (j = 0; j < windowSize; j++)
                job_array[j].alloc_total = 0;

	if (bidCtr == 0) {
		debug2("No bids formed, no IP solving necessary.");
	//	return 0;
	}
	
        alpha = minPrio / (bidCtr + 1);
        debug2("minPrio is %.2f, bidCtr is %d, alpha is %.2f",minPrio,bidCtr,alpha);
        
	debug2("timer: bid generation took %u",(uint16_t)(time(NULL)-time1));
	/* buradan oncesine yaziyorum herseyi */
                                                                                        	
        /*
	char envstr[256];
	sprintf(envstr,"ILOG_LICENSE_FILE=%s",get_cplex_license_address());
	if ( envstr != NULL ) {
		CPXputenv (envstr);
	}
	*/

	env = CPXopenCPLEX (&status);
	if ( env == NULL ) {
		char  errmsg[1024];
		CPXgeterrorstring (env, status, errmsg);
		fatal ("Could not open CPLEX environment. Error: %s",errmsg);
		goto TERMINATE;
	}

	lp = CPXcreateprob (env, &status, "lpex1");

	if (lp == NULL) {
		fatal("Failed to create LP.");
		goto TERMINATE;
	}

	time1 = time(NULL);
	status = populatebynonzero (env, lp, m, n, node_array, job_array);
	debug2("timer: populating lp took %u",(uint16_t)(time(NULL)-time1));
	debug2("returned from populate %d", status);
	sprintf(filename,"/root/logs/instance.lp");
	debug2("lp file: %s",filename);
	CPXwriteprob (env, lp, filename, NULL);
	debug2("timer: writing lp took %u",(uint16_t)(time(NULL)-time1));

	if ( status ) {
		fatal("Failed to populate problem.");
		goto TERMINATE;
	}		
	status = CPXsetintparam(env, CPX_PARAM_PARALLELMODE, CPX_PARALLEL_OPPORTUNISTIC);
	
	status = CPXsetdblparam(env, CPX_PARAM_TILIM, timeout);
	debug2("time limit set to %d", timeout);
	if ( status ) {
		fatal("Time limit problem.");
	}

	status = CPXsetintparam(env, CPX_PARAM_THREADS, 12);
	debug2("threads set to %d", 6);
	if ( status ) {
		fatal("thread setting problem.");
	}

	time1 = time(NULL);
	debug("optimizing");
	status = CPXmipopt (env, lp);
	debug2("timer: solving ip took %u",(uint16_t)(time(NULL)-time1));

	if (status == CPXERR_NO_SOLN) {
	        debug("No solution exists. Probably window size error. Reducing window size, status: %d", status);
	        half = true;
	}
	
	if ((status == CPXMIP_TIME_LIM_INFEAS) || 
		((status == CPXMIP_TIME_LIM_FEAS) && (objval < 0.1)) ) {
		debug("Time limit hit, reducing window size by half, status: %d.",status);
		half = true;
	} 
	

	if ( status ) {
		debug("Failed to optimize LP. status: %d",status);
		goto TERMINATE;
	}
	
	status = CPXgetobjval (env, lp, &objval); 
	if ( status ) {
		debug("Error in objective value: %d.",status);
		half = true;
		goto TERMINATE;
	}
	
	debug2("objective value: %.2f",objval);
	if ( (objval < 0.01) && ((int)(time(NULL)-time1) >= timeout) ) {
	        debug("Objective value 0 and time limit hit. Reducing window size.");
	        return 2;
	}

	status = CPXgetstat (env, lp);
	
	cur_numrows = CPXgetnumrows (env, lp);
	cur_numcols = CPXgetnumcols (env, lp);
	debug2("numrows: %d, numcols: %d",cur_numrows,cur_numcols);
	x = (double *) malloc (cur_numcols * sizeof(double));

	if ( x == NULL ) {
		status = CPXERR_NO_MEMORY;
		fatal ("Could not allocate memory for solution.");
		goto TERMINATE;
	}

	status = CPXgetmipx (env, lp, x, 0, cur_numcols - 1);
	
	if ( status ) {
		fatal ("Failed to get optimal integer x, status: %d", status);
		goto TERMINATE;
	}
	
	log = fopen("/root/logs/log","a+");
	fprintf(log,"size %d, begin %d, mid: %d, end %d, diff1 %d diff2 %d numBids %d filename %s\n",
		n, (int)begin_time, (int)mid_time, 
		(int)time(NULL), (int)(time(NULL)-begin_time),
		(int)(time(NULL) - mid_time), bidCtr, filename);
	fclose(log);
	
        sprintf(filename,"/root/logs/out.%d",mid_time);
	log = fopen(filename,"w");
        for (k = 0; k < cur_numcols; k++) {
                fprintf(log,"x%d: %.2f\n",k,x[k]);
        }
        fclose(log);
	
	if (! status) {
 	for (k = 0; k < bidCtr; k++) {
		if (x[k] > 0.99) {
			/* j'th bid is selected, which job does it belong to ? */
			j = bids[k].jobIdx;
			dummy = 0;
			sjob_ptr = &job_array[j];
			debug2("cplex allocated job %d, bid %d, x: %.2f",sjob_ptr->job_id,k,x[k]);
			job_det_ptr = sjob_ptr->job_ptr->details;
			sjob_ptr->node_bitmap = (bitstr_t *) 
				bit_alloc (node_record_count);
			job_det_ptr->req_node_bitmap = (bitstr_t *) 
				bit_alloc (node_record_count);
			job_det_ptr->req_node_layout = (uint16_t *) xmalloc 
				(sizeof(uint16_t) * node_record_count);
			job_det_ptr->req_node_bitmap = (bitstr_t *) bit_alloc
				(node_record_count);
                        debug2("job %d has %d nodes",sjob_ptr->job_id,bids[k].lastNode - bids[k].firstNode);
			for (c = bids[k].firstNode; c < bids[k].lastNode; c++) {
				i = jobNodeName[c];
				debug3("setting node %d (c: %d) for job %d",i,c, sjob_ptr->job_id);
				bit_set (sjob_ptr->node_bitmap, (bitoff_t) (i));
				bit_set (job_det_ptr->req_node_bitmap, (bitoff_t) (i));		
				job_det_ptr->req_node_layout[i] = 1 + (uint16_t) (x[bidCtr + jobNodeIdx[c] + jobNodeCtr]);
				if (x[bidCtr + jobNodeIdx[c] + jobNodeCtr] - floor(x[bidCtr + jobNodeIdx[c] + jobNodeCtr]) > 0.9) {
				        dummy = dummy + 1 + (uint16_t) ceil(x[bidCtr + jobNodeIdx[c] + jobNodeCtr]);
                                        job_det_ptr->req_node_layout[i] = 1 + (uint16_t) ceil(x[bidCtr + jobNodeIdx[c] + jobNodeCtr]);
                                }
                                else {
   				        dummy = dummy + 1 + (uint16_t) floor(x[bidCtr + jobNodeIdx[c] + jobNodeCtr]);
				        job_det_ptr->req_node_layout[i] = 1 + (uint16_t) floor(x[bidCtr + jobNodeIdx[c] + jobNodeCtr]);
                                }
                                debug3("set node above");

                                
				//debug3("job %d has allocation in node %d : %.3f (u %d r %d) %u node rem: %d alloc now: %d",sjob_ptr->job_id,i,
				        //(x[bidCtr + jobNodeIdx[c]]) + ceil(x[bidCtr + jobNodeIdx[c] + jobNodeCtr]),
				        //(uint16_t) (x[bidCtr + jobNodeIdx[c]]), (uint16_t) (x[bidCtr + jobNodeCtr + jobNodeIdx[c]]),
					//job_det_ptr->req_node_layout[i],node_array[i].rem_cpus,dummy);
			} 
			sjob_ptr->alloc_total = dummy;
			debug3("set alloc total to %d",dummy);
		} 
	} 
	}
	
TERMINATE:
        debug3("moved to terminate");

	free(intervals);
	debug3("freed intervals");
	free(cumsum);
	debug3("freed cumsum");
	for (i = 0; i < classCtr; i++)
	        free(cArray[i].nodesUsed);
        debug3("freed nodesused");
        free(cArray);
        debug3("freed cArray");
        free(bids);
        debug3("freed bids");
        free(jobNodeName);
        free(jobNodeIdx);
        debug3("freed jobNodeNameIdx");
        
	free_and_null ((char **) &x);
	free_and_null ((char **) &slack);
	free_and_null ((char **) &dj);
	free_and_null ((char **) &pi);

	if (lp != NULL) {
		status2 = CPXfreeprob (env, &lp);
		if (status2) {
			fatal("CPXfreeprob failed, error code %d.", status2);
		}
	}

	if (env != NULL) {
		status2 = CPXcloseCPLEX (&env);
		if (status2) {
			char errmsg[1024];
			fatal("Could not close CPLEX environment.");
			CPXgeterrorstring (env, status2, errmsg);
			fatal("%s", errmsg);
		}
	}     
	
	if (half)
	        return 2;
	
	return (status);
}

static void free_and_null (char **ptr)
{
	if ( *ptr != NULL ) {
		free (*ptr);
		*ptr = NULL;
	}
}
