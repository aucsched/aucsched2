# AUCSCHED2 scheduler plug-in for Slurm
auction based scheduler with gpu ranges support (released version)

Developed by Seren Soner and Can Ozturan from Bogazici University, Istanbul, Turkey

Contact seren.soner@gmail.com, ozturaca@boun.edu.tr if necessary

* IMPORTANT NOTE: sched currently works with only CR_CPU with GresTypes=gpu (optional)
  nodes should be SHARED, overcommit is not allowed.

* CPLEX REQUIRED: after running patch.sh, set ILOG dir variables in sched/plugins/aucsched/Makefile.am and 
  sched/plugins/aucsched/Makefile.in 

Main idea
----------

* from the queue, do not start the first job, but take a set of jobs 
* test all of them at once to see selection of which of these jobs 
  would yield the best result

IMPORTANT NOTE ABOUT PATCHES
----------------------------
* patches on gres.c and gres.h add two functions, which return number of gres types defined, and the number of gpu's a specific job requests. also, one function is modified to return number of gpu's available in that node, instead of not returning anything.
* patch on job_scheduler.c changes the job scheduler so that, when a job is submitted, its not immediately considered by the job scheduler; but waits for the next run of IP solve to be allocated.
* patch on configure, makefile.am and makefile.in files allow the makefiles to be generated automatically.



