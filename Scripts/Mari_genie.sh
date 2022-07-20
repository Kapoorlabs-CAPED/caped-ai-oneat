#!/bin/bash
JID_JOB1=`sbatch  jobscript_split | cut -d " " -f 4`
JID_JOB2=`sbatch  --dependency=afterok:$JID_JOB1 jobscript | cut -d " " -f 4`
JID_JOB3=`sbatch  --dependency=afterok:$JID_JOB2 jobscript_merge | cut -d " " -f 4`
sbatch  --dependency=afterok:$JID_JOB3 jobscriptoneat
