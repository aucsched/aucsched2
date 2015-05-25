#/bin/bash
cp *patch ..
cp -r plugins/sched/aucsched ../src/plugins/sched/
cp -r plugins/select/lpconsres ../src/plugins/select/
cd ..
patch -p0 < configure.ac.patch
patch -p0 < configure.patch
patch -p0 < gres.c.patch
patch -p0 < gres.h.patch
patch -p0 < job_scheduler.patch
patch -p0 < schedMakefile.am.patch
patch -p0 < schedMakefile.in.patch
patch -p0 < selectMakefile.am.patch
patch -p0 < selectMakefile.in.patch
rm -f configure.ac.patch gres.c.patch job_scheduler.patch schedMakefile.in.patch selectMakefile.in.patch configure.patch gres.h.patch schedMakefile.am.patch selectMakefile.am.patch

echo "modify src/plugins/sched/Makefile.am and Makefile.in and add your ILOG dir"

