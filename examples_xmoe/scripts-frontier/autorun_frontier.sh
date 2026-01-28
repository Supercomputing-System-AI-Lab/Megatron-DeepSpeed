#!/bin/bash

echo "Starting experiment submission process..."

##########################################################################
# --- 1. CONFIGURE YOUR EXPERIMENTS HERE ---
#
# We now use associative arrays (maps) to link configurations together.
# The KEY for each map is the node configuration: "NODES:TOTAL_GPUS"
# The VALUE is a space-separated string of the settings for that key.
##########################################################################

# Set the Tensor Parallelism size (usually constant)
MP_SIZE=1

# --- A) PIPELINE PARALLEL (PP) CONFIGURATIONS ---
# (Example: PP_STRATEGY_MAP["4:32"]="4:8 8:4") PP:EP
# (Example: PP_BATCH_MAP["4:32"]="4:256:100:X-MOE:10.1")
declare -A PP_STRATEGY_MAP
PP_STRATEGY_MAP["1:2"]="2:1" 
PP_STRATEGY_MAP["1:4"]="4:1" 
PP_STRATEGY_MAP["1:8"]="2:4" 
PP_STRATEGY_MAP["2:8"]="2:4" 
PP_STRATEGY_MAP["2:16"]="2:8 4:4" 
PP_STRATEGY_MAP["4:16"]="4:4" 
PP_STRATEGY_MAP["4:32"]="4:8" 
PP_STRATEGY_MAP["5:40"]="5:8" 
PP_STRATEGY_MAP["6:48"]="6:8" 
PP_STRATEGY_MAP["8:64"]="8:8" #"4:8" 
PP_STRATEGY_MAP["12:96"]="6:8" #"12:8" 
PP_STRATEGY_MAP["15:120"]="15:8" 
PP_STRATEGY_MAP["16:128"]="4:8" #"16:8" 
PP_STRATEGY_MAP["20:160"]="20:8" 
PP_STRATEGY_MAP["24:192"]="4:8" #"24:8" 
PP_STRATEGY_MAP["30:240"]="30:8" 
PP_STRATEGY_MAP["32:256"]="4:8" #"32:8" 
PP_STRATEGY_MAP["40:320"]="40:8" 
PP_STRATEGY_MAP["60:480"]="60:8" 
PP_STRATEGY_MAP["64:512"]="4:8" #"32:8" 
PP_STRATEGY_MAP["128:1024"]="4:8" #"32:8" 


declare -A PP_BATCH_MAP

# PP_BATCH_MAP["1:8"]="8:200:20:X-MOE:10B:no-ckpt:1:even:no-planner 7:200:20:X-MOE:10B:no-ckpt:1:even:no-planner 6:200:20:X-MOE:10B:no-ckpt:1:even:no-planner 5:200:20:X-MOE:10B:no-ckpt:1:even:no-planner  4:200:20:X-MOE:10B:no-ckpt:1:even:no-planner "
# PP_BATCH_MAP["2:16"]="2:100:20:X-MOE:10B:no-ckpt:1:even:no-planner 4:100:20:X-MOE:10B:no-ckpt:1:even:no-planner "

# PP_BATCH_MAP["1:4"]="1:100:20:X-MOE:10B:no-ckpt:1:even:no-planner 2:100:20:X-MOE:10B:no-ckpt:1:even:no-planner 4:100:20:X-MOE:10B:no-ckpt:1:even:no-planner 6:100:20:X-MOE:10B:no-ckpt:1:even:no-planner 8:100:20:X-MOE:10B:no-ckpt:1:even:no-planner 10:100:20:X-MOE:10B:no-ckpt:1:even:no-planner"
# PP_BATCH_MAP["1:8"]="1:100:20:X-MOE:10B:ckpt:1:even:no-planner"
# PP_BATCH_MAP["1:4"]="1:100:20:X-MOE:10B:ckpt:1:even:no-planner"
# PP_BATCH_MAP["1:2"]="1:100:20:X-MOE:10B:ckpt:1:even:no-planner"

# PP_BATCH_MAP["1:8"]="1:10:20:X-MOE:10B:no-ckpt:1:even:no-planner"
# PP_BATCH_MAP["60:480"]="1:1920:14:X-MOE:537B:ckpt:1 1:1920:14:X-MOE:537B:dynamic-ckpt:1  "
# PP_BATCH_MAP["4:32"]="1:96:3:X-MOE:50B:ckpt:1:even 1:96:3:X-MOE:50B:dynamic-ckpt:1:uneven 1:96:3:X-MOE:50B:dynamic-ckpt:1:even"
# PP_BATCH_MAP["1:8"]="1:960:20:X-MOE:10B:ckpt:1:uneven 1:960:20:X-MOE:10B:dynamic-ckpt:1:even 1:960:20:X-MOE:10B:dynamic-ckpt:1:uneven 1:960:20:X-MOE:10B:ckpt:1:even 1:960:20:X-MOE:10B:no-ckpt:1:even "
# PP_BATCH_MAP["4:32"]="1:960:20:X-MOE:50B:ckpt:1:uneven 1:960:20:X-MOE:50B:dynamic-ckpt:1:even 1:960:20:X-MOE:50B:dynamic-ckpt:1:uneven 1:960:20:X-MOE:50B:ckpt:1:even 1:960:20:X-MOE:50B:no-ckpt:1:even "

# PP_BATCH_MAP["30:240"]="1:200:16:X-MOE:537B:ckpt:1:even:no-planner 1:200:16:X-MOE:537B:dynamic-ckpt:1:uneven:yes-planner"
# PP_BATCH_MAP["30:240"]="1:200:16:X-MOE:537B:dynamic-ckpt:1:uneven:yes-planner"
# PP_BATCH_MAP["40:320"]="1:100:16:X-MOE:537B:dynamic-ckpt:1:uneven:yes-planner"
# PP_BATCH_MAP["60:480"]="1:200:15:X-MOE:537B:ckpt:1:even:no-planner 1:230:16:X-MOE:1T:ckpt:1:even:no-planner"
# PP_BATCH_MAP["60:480"]="1:230:16:X-MOE:1T:ckpt:1:even:no-planner 1:230:16:X-MOE:1T:dynamic-ckpt:1:uneven:yes-planner "
# PP_BATCH_MAP["60:480"]="1:400:16:X-MOE:1T:dynamic-ckpt:1:uneven:yes-planner " # 01/16/2026 rerun for 2h with increase gbs
# PP_BATCH_MAP["1:8"]="2:4:4:X-MOE:10B:dynamic-ckpt:1:uneven:yes-planner"
# PP_BATCH_MAP["1:8"]="2:4:4:X-MOE:10B:dynamic-ckpt:1:uneven:no-planner"
# PP_BATCH_MAP["4:32"]="4:20:20:X-MOE:10B:no-ckpt:1:even:no-planner"

# PP_BATCH_MAP["4:32"]="5:200:20:X-MOE:10B:no-ckpt:0:even:no-planner 6:200:20:X-MOE:10B:no-ckpt:0:even:no-planner 7:200:20:X-MOE:10B:no-ckpt:0:even:no-planner 8:200:20:X-MOE:10B:no-ckpt:0:even:no-planner 9:200:20:X-MOE:10B:no-ckpt:0:even:no-planner 10:200:20:X-MOE:10B:no-ckpt:0:even:no-planner  "
# PP_BATCH_MAP["4:32"]="1:200:20:X-MOE:10B:yes-ckpt:1:uneven:yes-planner   "
# PP_BATCH_MAP["4:32"]="1:200:30:X-MOE:63B:yes-ckpt:1:even:no-planner 1:200:30:X-MOE:63B:dynamic-ckpt:1:uneven:yes-planner " # 01/16/2026 rerun
# PP_BATCH_MAP["4:32"]="1:200:22:X-MOE:63B:dynamic-ckpt:1:uneven:yes-planner " # 01/16/2026 rerun
# PP_BATCH_MAP["16:128"]="1:200:20:X-MOE:173B:yes-ckpt:1:even:no-planner 1:200:20:X-MOE:173B:dynamic-ckpt:1:uneven:yes-planner "
# PP_BATCH_MAP["12:96"]="1:200:20:X-MOE:173B:yes-ckpt:1:even:no-planner"
# PP_BATCH_MAP["12:96"]="1:200:20:X-MOE:173B:dynamic-ckpt:1:uneven:yes-planner "

# 01/17/2026 strong scaling (fix optimal mbs while varying nms to control fixed gbs)
# PP_BATCH_MAP["4:32"]="1:800:10:X-MOE:63B:dynamic-ckpt:1:uneven:yes-planner  " 
# PP_BATCH_MAP["8:64"]="1:400:10:X-MOE:63B:dynamic-ckpt:1:uneven:yes-planner " 
# PP_BATCH_MAP["16:128"]="1:200:10:X-MOE:63B:dynamic-ckpt:1:uneven:yes-planner " 
# PP_BATCH_MAP["24:192"]="1:133:10:X-MOE:63B:dynamic-ckpt:1:uneven:yes-planner " 
# PP_BATCH_MAP["32:256"]="1:100:10:X-MOE:63B:dynamic-ckpt:1:uneven:yes-planner " 

# PP_BATCH_MAP["16:128"]="1:1200:7:X-MOE:63B:dynamic-ckpt:1:uneven:yes-planner " 
# PP_BATCH_MAP["32:256"]="1:600:7:X-MOE:63B:dynamic-ckpt:1:uneven:yes-planner " 
# PP_BATCH_MAP["64:512"]="1:300:7:X-MOE:63B:dynamic-ckpt:1:uneven:yes-planner " 
# PP_BATCH_MAP["128:1024"]="1:150:7:X-MOE:63B:dynamic-ckpt:1:uneven:yes-planner " 

# # 01/17/2026 weak scaling (fix optimal mbs while scaling nms to scale gbs)
# PP_BATCH_MAP["4:32"]="1:100:10:X-MOE:63B:dynamic-ckpt:1:uneven:yes-planner  " 
# PP_BATCH_MAP["8:64"]="1:100:10:X-MOE:63B:dynamic-ckpt:1:uneven:yes-planner " 
# PP_BATCH_MAP["16:128"]="1:100:10:X-MOE:63B:dynamic-ckpt:1:uneven:yes-planner " 
# PP_BATCH_MAP["24:192"]="1:100:10:X-MOE:63B:dynamic-ckpt:1:uneven:yes-planner " 
# PP_BATCH_MAP["32:256"]="1:100:10:X-MOE:63B:dynamic-ckpt:1:uneven:yes-planner " 


# #5 pp+act_ckpt on tflops, mem, mbs (#6 too)
# PP_BATCH_MAP["8:64"]="1:50:10:X-MOE:63B:ckpt:1:even:no-planner 2:50:10:X-MOE:63B:ckpt:1:even:no-planner 4:50:10:X-MOE:63B:ckpt:1:even:no-planner 6:50:10:X-MOE:63B:ckpt:1:even:no-planner 8:50:10:X-MOE:63B:ckpt:1:even:no-planner        1:50:10:X-MOE:63B:no-ckpt:0:even:no-planner 2:50:10:X-MOE:63B:no-ckpt:0:even:no-planner 4:50:10:X-MOE:63B:no-ckpt:0:even:no-planner" 
# PP_BATCH_MAP["8:64"]=" 2:50:12:X-MOE:63B:ckpt:1:even:no-planner " 
# PP_BATCH_MAP["8:64"]=" 4:50:12:X-MOE:63B:ckpt:1:even:no-planner " 


# #8 dynamic checkpointing
# PP_BATCH_MAP["4:32"]=" 1:100:15:X-MOE:63B:ckpt:1:even:no-planner " 
# PP_BATCH_MAP["4:32"]=" 4:100:12:X-MOE:63B:ckpt:1:even:no-planner " 
# PP_BATCH_MAP["4:32"]=" 2:100:15:X-MOE:63B:ckpt:1:even:no-planner " 

# PP_BATCH_MAP["4:32"]="1:100:20:X-MOE:63B:dynamic-ckpt:1:uneven:yes-planner  " 
# PP_BATCH_MAP["8:64"]="1:200:20:X-MOE:63B:dynamic-ckpt:1:uneven:yes-planner " 
# PP_BATCH_MAP["16:128"]="1:400:20:X-MOE:63B:dynamic-ckpt:1:uneven:yes-planner " 
# PP_BATCH_MAP["32:256"]="1:800:20:X-MOE:63B:dynamic-ckpt:1:uneven:yes-planner " 

# "1:96:5:X-MOE:50B:ckpt:1:even:no-planner"
# "1:96:5:X-MOE:50B:ckpt:1:even:planner-small"
# "1:96:5:X-MOE:50B:ckpt:1:even:planner-medium"
# "1:96:5:X-MOE:50B:ckpt:1:even:planner-large"

# check planner throughput 
# PP_BATCH_MAP["4:32"]="   4:1280:14:X-MOE:63B:ckpt:1:even   3:960:16:X-MOE:63B:ckpt:1:even     "
# PP_BATCH_MAP["4:32"]="  1:720:20:X-MOE:63B:dynamic-ckpt:1:uneven   1:720:20:X-MOE:63B:ckpt:1:even  2:1440:13:X-MOE:63B:ckpt:1:even "
# PP_BATCH_MAP["4:32"]="7:1120:20:X-MOE:50B:ckpt:1:uneven 8:1280:20:X-MOE:50B:ckpt:1:uneven 6:960:20:X-MOE:50B:ckpt:1:even"
# PP_BATCH_MAP["4:32"]="1:100:15:X-MOE:63B:dynamic-ckpt:1:uneven:yes-planner "
# PP_BATCH_MAP["4:32"]="1:100:15:X-MOE:63B:dynamic-ckpt:1:even:no-planner "

# PP_BATCH_MAP["1:8"]="1:96:3:X-MOE:10B:dynamic-ckpt:1:uneven"
# PP_BATCH_MAP["1:4"]="1:8:2:X-MOE:10B:ckpt:1"

# PP_BATCH_MAP["4:32"]="1:1152:20:X-MOE:50B:no-ckpt"



# --- B) EXPERT PARALLEL (EP) CONFIGURATIONS ---
# This example is adapted from your latest slurm script.
# KEY: NODE:TOTAL_GPU
# VALUE="BS:NBS:TRAIN_ITERS:MOE_TYPE:MODEL_SIZE"
declare -A EP_BATCH_MAP
# EP_BATCH_MAP["1:8:1"]="4:30:20:X-MOE:10B:no-ckpt:0 5:30:20:X-MOE:10B:no-ckpt:0 "

# #4 1 node ep tflops, mem, mbs, ckpt vs no-ckpt
# EP_BATCH_MAP["1:8:1"]="1:10:20:X-MOE:10B:no-ckpt:0 2:10:20:X-MOE:10B:no-ckpt:0 4:10:20:X-MOE:10B:no-ckpt:0 6:10:20:X-MOE:10B:no-ckpt:0 8:10:20:X-MOE:10B:no-ckpt:0              1:10:20:X-MOE:10B:ckpt:1 2:10:20:X-MOE:10B:ckpt:1 4:10:20:X-MOE:10B:ckpt:1 6:10:20:X-MOE:10B:ckpt:1 8:10:20:X-MOE:10B:ckpt:1 "
# EP_BATCH_MAP["1:8:1"]=" 12:10:20:X-MOE:10B:ckpt:1 16:10:20:X-MOE:10B:ckpt:1 20:10:20:X-MOE:10B:ckpt:1 "
# EP_BATCH_MAP["2:16:1"]="2:10:20:X-MOE:10B:no-ckpt:0 4:10:20:X-MOE:10B:no-ckpt:0 6:10:20:X-MOE:10B:no-ckpt:0 8:10:20:X-MOE:10B:no-ckpt:0"


# weak scaling 
# EP_BATCH_MAP["4:32:1"]="1:25:10:X-MOE:63B:ckpt:1  " 
# EP_BATCH_MAP["8:32:1"]="1:25:10:X-MOE:63B:ckpt:1  " 
# EP_BATCH_MAP["16:32:1"]="1:25:10:X-MOE:63B:ckpt:1  " 
# EP_BATCH_MAP["24:32:1"]="1:25:10:X-MOE:63B:ckpt:1  " 
# EP_BATCH_MAP["32:32:1"]="1:25:10:X-MOE:63B:ckpt:1  " 

# # # strong scaling 
# EP_BATCH_MAP["4:32:1"]="1:200:9:X-MOE:63B:ckpt:1  " 
# EP_BATCH_MAP["8:32:1"]="1:100:10:X-MOE:63B:ckpt:1  " 
# EP_BATCH_MAP["16:32:1"]="1:50:10:X-MOE:63B:ckpt:1  " 
# EP_BATCH_MAP["24:32:1"]="1:37:10:X-MOE:63B:ckpt:1  " 
# EP_BATCH_MAP["32:32:1"]="1:25:10:X-MOE:63B:ckpt:1  " 


# EP_BATCH_MAP["4:32"]="1:64:20:X-MOE:50B "
# EP_BATCH_MAP["8:64"]="1:512:20:X-MOE:50B 2:512:20:X-MOE:50B 2:1024:20:X-MOE:50B"
# EP_BATCH_MAP["8:64"]="1:64:15:DS-MOE:10B 2:128:15:DS-MOE:10B 3:192:15:DS-MOE:10B"
# EP_BATCH_MAP["16:128:2"]="1:2048:30:X-MOE:190B"

# EP_BATCH_MAP["1:8:1"]="  1:4:20:X-MOE:173B_2L:no-ckpt:0 "

# # #7, profiling A2A time across 1,2,4,8 nodes
# EP_BATCH_MAP["1:8:1"]=" 1:4:15:X-MOE:63B_4L:no-ckpt:0  2:4:15:X-MOE:63B_4L:no-ckpt:0  4:4:15:X-MOE:63B_4L:no-ckpt:0  8:4:15:X-MOE:63B_4L:no-ckpt:0  16:4:15:X-MOE:63B_4L:no-ckpt:0  "
# EP_BATCH_MAP["2:16:1"]=" 1:4:15:X-MOE:63B_4L:no-ckpt:0  2:4:15:X-MOE:63B_4L:no-ckpt:0  4:4:15:X-MOE:63B_4L:no-ckpt:0  8:4:15:X-MOE:63B_4L:no-ckpt:0   16:4:15:X-MOE:63B_4L:no-ckpt:0 "
# EP_BATCH_MAP["4:32:1"]=" 1:4:15:X-MOE:63B_4L:no-ckpt:0  2:4:15:X-MOE:63B_4L:no-ckpt:0  4:4:15:X-MOE:63B_4L:no-ckpt:0  8:4:15:X-MOE:63B_4L:no-ckpt:0   16:4:15:X-MOE:63B_4L:no-ckpt:0 "
# EP_BATCH_MAP["8:64:1"]=" 1:4:15:X-MOE:63B_4L:no-ckpt:0  2:4:15:X-MOE:63B_4L:no-ckpt:0  4:4:15:X-MOE:63B_4L:no-ckpt:0  8:4:15:X-MOE:63B_4L:no-ckpt:0   16:4:15:X-MOE:63B_4L:no-ckpt:0 "
# EP_BATCH_MAP["8:64:1"]=" 2:4:15:X-MOE:63B_4L:no-ckpt:0   "

# EP_BATCH_MAP["64:512:4"]="  1:10:16:X-MOE:537B:ckpt:1  "
# EP_BATCH_MAP["128:1024:4"]=" 1:7:16:X-MOE:537B:no-ckpt:0  1:7:16:X-MOE:537B:ckpt:1  1:7:16:X-MOE:1T:ckpt:1  "
# EP_BATCH_MAP["128:1024:4"]=" 1:10:16:X-MOE:1T:ckpt:1  "

# EP_BATCH_MAP["32:64:4"]=" 1:10:30:X-MOE:173B:no-ckpt:0  "
# EP_BATCH_MAP["16:128:1"]="  1:10:30:X-MOE:173B:yes-ckpt:1 "
# EP_BATCH_MAP["8:64:1"]=" 1:10:30:X-MOE:63B:no-ckpt:0 1:10:30:X-MOE:63B:yes-ckpt:1 "
# EP_BATCH_MAP["4:32:1"]=" 1:10:30:X-MOE:63B:yes-ckpt:1 "
# EP_BATCH_MAP["32:256:1"]=" 1:10:30:X-MOE:173B:ckpt:1 " #     3964070  
# EP_BATCH_MAP["32:256:4"]=" 1:6:16:X-MOE:173B:no-ckpt:0 1:10:16:X-MOE:173B:ckpt:1 " #     01/13/2026 172B xmoe rerun  
# EP_BATCH_MAP["32:256:4"]=" 1:6:16:X-MOE:173B:no-ckpt:0 " #     01/16/2026 172B xmoe rerun  
# EP_BATCH_MAP["32:256:2"]=" 1:10:20:X-MOE:173B:no-ckpt:0 " #     01/16/2026 172B xmoe rerun  
# EP_BATCH_MAP["32:256:4"]=" 1:10:25:X-MOE:173B:no-ckpt:0 " #     01/13/2026 172B xmoe rerun  
# EP_BATCH_MAP["8:64:1"]=" 1:20:25:X-MOE:63B:no-ckpt:0 " # 01/16/2026 63B rerun
# EP_BATCH_MAP["16:128:1"]="  1:7:12:X-MOE:173B:ckpt:1 " #  3964072  
# EP_BATCH_MAP["8:64:1"]=" 1:10:30:X-MOE:63B:ckpt:1 "
# EP_BATCH_MAP["4:32:1"]=" 1:10:30:X-MOE:63B:ckpt:1 " #  3964073  

# EP_BATCH_MAP["8:64:1"]=" 1:20:20:DS-MOE:63B:no-ckpt:0  1:20:25:DS-MOE:63B:ckpt:1 " # 01/22/2026 63B DS-MoE rerun
# EP_BATCH_MAP["32:256:2"]=" 1:20:15:DS-MOE:173B:no-ckpt:0  1:20:15:DS-MOE:173B:ckpt:1 " #     01/22/2026 172B DS-moe rerun  
# EP_BATCH_MAP["32:256:1"]=" 1:7:16:DS-MOE:537B:no-ckpt:0  1:7:16:DS-MOE:537B:ckpt:1 1:20:15:DS-MOE:173B:no-ckpt:0  1:20:15:DS-MOE:173B:ckpt:1  " #   01/22/2026 537B DS-moe rerun  
# EP_BATCH_MAP["32:256:1"]=" 1:7:16:DS-MOE:537B:no-ckpt:0   " #   01/22/2026 537B DS-moe rerun  
# EP_BATCH_MAP["32:256:1"]=" 1:7:16:DS-MOE:537B:no-ckpt:0   " #   01/22/2026 537B DS-moe rerun  


# 01/23/2026: Baseline runs 
# EP_BATCH_MAP["8:64:1"]=" 1:20:12:DS-MOE:63B:no-ckpt:0  1:20:12:DS-MOE:63B:ckpt:1 " # 01/22/2026 63B DS-MoE rerun
# EP_BATCH_MAP["8:64:2"]="  1:20:12:TUTEL-MOE:63B:no-ckpt:0  1:20:12:TUTEL-MOE:63B:ckpt:1 " # 01/23/2026 63B DS-MoE rerun
# EP_BATCH_MAP["8:32:2"]=" 1:20:12:TED-MOE:63B:no-ckpt:0  1:20:12:TED-MOE:63B:ckpt:1 "

# EP_BATCH_MAP["32:128:1"]=" 1:20:13:DS-MOE:173B:no-ckpt:0  1:20:13:DS-MOE:173B:ckpt:1 " # 01/22/2026 63B DS-MoE rerun
# EP_BATCH_MAP["32:128:2"]=" 1:20:13:TED-MOE:173B:no-ckpt:0  1:20:13:TED-MOE:173B:ckpt:1   1:20:13:TUTEL-MOE:173B:no-ckpt:0  1:20:13:TUTEL-MOE:173B:ckpt:1 " # 01/23/2026 63B DS-MoE rerun
# EP_BATCH_MAP["32:128:2"]=" 1:20:13:TED-MOE:173B:no-ckpt:0  1:20:13:TED-MOE:173B:ckpt:1 "

EP_BATCH_MAP["64:256:1"]=" 1:5:2:DS-MOE:537B:ckpt:1 " # 01/22/2026 63B DS-MoE rerun
EP_BATCH_MAP["64:256:2"]="  1:5:2:TED-MOE:537B:ckpt:1   1:5:2:TUTEL-MOE:537B:ckpt:1 " # 01/23/2026 63B DS-MoE rerun


# EP_BATCH_MAP["4:32:1"]="2:20:25:X-MOE:10B:no-ckpt:0 4:20:25:X-MOE:10B:no-ckpt:0 6:20:25:X-MOE:10B:no-ckpt:0 8:20:25:X-MOE:10B:no-ckpt:0 10:20:25:X-MOE:10B:no-ckpt:0 "



# EP_BATCH_MAP["1:8:1"]=" 1:3:15:X-MOE:173B_1L:no-ckpt:0  "
# EP_BATCH_MAP["4:32:1"]=" 1:1024:15:X-MOE:50B:no-ckpt:0   1:1024:15:X-MOE:50B:ckpt:1 2:1024:15:X-MOE:50B:ckpt:1 4:1024:15:X-MOE:50B:ckpt:1    "
# EP_BATCH_MAP["16:128:2"]="1:1024:15:X-MOE:190B:no-ckpt:0    1:1024:15:X-MOE:190B:ckpt:1 2:1024:15:X-MOE:190B:ckpt:1 4:1024:15:X-MOE:190B:ckpt:1   "
# EP_BATCH_MAP["4:32:1"]="1:96:5:X-MOE:50B:no-ckpt:0"


# EP_BATCH_MAP["1:8:1"]="1:24:2:X-MOE:10B:no-ckpt:0 "

# EP_BATCH_MAP["2:16"]="1:64:4:X-MOE:10B"

# EP_BATCH_MAP["2:16"]="1:96:30:X-MOE:10B"
# EP_BATCH_MAP["1:8"]="1:96:20:X-MOE:10B "
# EP_BATCH_MAP["12:96"]="1:96:30:X-MOE:50B"
# EP_BATCH_MAP["4:32"]="1:96:30:X-MOE:190B_div_4 1:96:30:X-MOE:10B"
# Add other EP configurations here, for example:
# EP_BATCH_MAP["2:16"]="1:128:30:X-MOE:10.1 2:128:30:X-MOE:10.1"
# EP_BATCH_MAP["4:32"]="1:96:30:X-MOE:50B"


declare -A PROFILE_MAP
# PROFILE_MAP["1:8:1"]=" 1:3:15:X-MOE:10B_1L:no-ckpt:0  "
# PROFILE_MAP["1:8:1"]=" 1:3:15:X-MOE:50B_1L:no-ckpt:0  "
# PROFILE_MAP["1:8:1"]=" 1:3:15:X-MOE:173B_1L:no-ckpt:0  "
# PROFILE_MAP["1:8:1"]=" 1:3:15:X-MOE:537B_1L:no-ckpt:0  "
# PROFILE_MAP["1:8:1"]=" 1:4:20:X-MOE:10B_1L:no-ckpt:0 1:4:20:X-MOE:50B_1L:no-ckpt:0  1:4:20:X-MOE:173B_1L:no-ckpt:0  1:4:20:X-MOE:537B_1L:no-ckpt:0   1:4:20:X-MOE:1T_1L:no-ckpt:0 "
# PROFILE_MAP["2:16:1"]=" 1:3:15:X-MOE:537B_1L:no-ckpt:0  "
# PROFILE_MAP["1:8:1"]=" 1:3:15:X-MOE:1T_1L:no-ckpt:0  "
# PROFILE_MAP["2:16:1"]=" 1:3:15:X-MOE:1T_1L:no-ckpt:0  "






# ==============================================================================
# HELPER: SUBMIT PROFILING JOB
# ==============================================================================
declare -A SUBMITTED_PROFILES

submit_profile_job() {
    local NODE_KEY=$1      # e.g., "4:32"
    local BATCH_CONFIG=$2  # e.g., "1:64:15:X-MOE:50B..."
    
    # 1. Define Template
    local PROF_TEMPLATE="planner.slurm.template"
    if [ ! -f "$PROF_TEMPLATE" ]; then
        echo "ERROR: Profiling template '$PROF_TEMPLATE' not found." >&2
        return 1
    fi

    # 2. Parse Inputs
    IFS=':' read -r NODES TOTAL_GPUS <<< "$NODE_KEY"
    IFS=':' read -r BS NBS TRAIN_ITERS MOE_TYPE MODEL_SIZE CHECKPOINT CHECKPOINT_NUM_LAYERS PP_PARTITION PLANNER_MODE <<< "$BATCH_CONFIG"

    # 3. Enforce Profiling Settings
    # Profiling is always EP-only (PP=1), Short run, No Checkpoint
    local PP_SIZE=1
    local MP_SIZE=1
    local EP_PARALLEL_SIZE=$(( TOTAL_GPUS / MP_SIZE ))
    
    # Append _1L to model size if not present
    if [[ "$MODEL_SIZE" != *"_1L"* ]]; then
        MODEL_SIZE="${MODEL_SIZE}_1L"
    fi

    local RUN_TYPE="profile_ep"
    local TRAIN_ITERS=15
    local ACTIVATION_CHECKPOINT="false"

    local JOB_NAME="profile_${MODEL_SIZE}_n${NODES}"
    # local TEMP_SCRIPT="temp_profile_${JOB_NAME}.slurm"
    local TEMP_DIR="temp_slurm"
    local TEMP_SCRIPT="${TEMP_DIR}/temp_profile_${JOB_NAME}.slurm"

    # Create temp_slurm directory if it doesn't exist
    mkdir -p "${TEMP_DIR}"

    echo "  - Generating Profiling job: ${JOB_NAME}" >&2

    # 4. Generate Script using the Planner Template
    sed -e "s/{{RUN_TYPE}}/${RUN_TYPE}/g" \
        -e "s/{{NODES}}/${NODES}/g" \
        -e "s/{{TOTAL_GPUS}}/${TOTAL_GPUS}/g" \
        -e "s/{{PARTITION}}/batch/g" \
        -e "s/{{BATCH_SIZE}}/${BS}/g" \
        -e "s/{{NUM_BATCHES}}/1/g" \
        -e "s/{{PP_SIZE}}/${PP_SIZE}/g" \
        -e "s/{{EP_PARALLEL_SIZE}}/${EP_PARALLEL_SIZE}/g" \
        -e "s/{{MP_SIZE}}/${MP_SIZE}/g" \
        -e "s/{{TRAIN_ITERS}}/${TRAIN_ITERS}/g" \
        -e "s/{{MOE_TYPE}}/${MOE_TYPE}/g" \
        -e "s/{{MODEL_SIZE}}/${MODEL_SIZE}/g" \
        -e "s/{{ACTIVATION_CHECKPOINT}}/${ACTIVATION_CHECKPOINT}/g" \
        -e "s/{{CHECKPOINT_NUM_LAYERS}}/0/g" \
        -e "s/{{DYNAMIC_CHECKPOINT}}/False/g" \
        -e "s/{{UNEVEN_PP}}/False/g" \
        -e "s/{{PROFILING_ENABLED}}/true/g" \
        ${PROF_TEMPLATE} > ${TEMP_SCRIPT}

    # 5. Submit and return ONLY the Job ID
    local JOB_ID=$(sbatch ${TEMP_SCRIPT} | awk '{print $4}')
    echo "$JOB_ID"
    
    # rm ${TEMP_SCRIPT}
}





##########################################################################
# --- 2. SCRIPT LOGIC (No need to edit below) ---
##########################################################################
TEMPLATE_FILE="frontier_xmoe.slurm.template"

if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "ERROR: The template file '$TEMPLATE_FILE' was not found."
    exit 1
fi

# --- LAUNCH PIPELINE PARALLEL (PP) RUNS ---
echo
echo "------------------------------------------------"
echo "--- Submitting Pipeline Parallel (PP) jobs ---"
echo "------------------------------------------------"

# Iterate over the node configurations defined in the PP strategy map
for node_key in "${!PP_STRATEGY_MAP[@]}"; do
    IFS=':' read -r NODES TOTAL_GPUS <<< "$node_key"

    # Get the specific strategies and batch configs for this node setup
    pp_strategies_string=${PP_STRATEGY_MAP[$node_key]}
    batch_configs_string=${PP_BATCH_MAP[$node_key]}

    if [ -z "$batch_configs_string" ]; then
        echo "Warning: No batch configs found for PP node setup ${node_key}. Skipping."
        continue
    fi

    # Set partition for the new environment
    PARTITION="batch"

    echo "Found PP configurations for ${NODES} nodes, ${TOTAL_GPUS} GPUs..."

    for pp_strategy in $pp_strategies_string; do
        IFS=':' read -r PP_SIZE EP_PARALLEL_SIZE <<< "$pp_strategy"

        # Sanity Check
        # if (( PP_SIZE * EP_PARALLEL_SIZE * MP_SIZE != TOTAL_GPUS )); then
        #     echo "  - Invalid strategy: PP=${PP_SIZE}, EP=${EP_PARALLEL_SIZE} for ${TOTAL_GPUS} GPUs. Skipping."
        #     continue
        # fi
        REQUIRED_GPUS=$(( PP_SIZE * EP_PARALLEL_SIZE * MP_SIZE ))

        # Check if the number of required GPUs is a multiple of the total available GPUs
        if (( TOTAL_GPUS % REQUIRED_GPUS == 0 )); then
            echo "Required GPUs (${REQUIRED_GPUS}) is divisible by Total GPUs (${TOTAL_GPUS})."
        else
            echo "Required GPUs (${REQUIRED_GPUS}) is NOT divisible by Total GPUs (${TOTAL_GPUS})."
        fi

        for batch_config in $batch_configs_string; do
            IFS=':' read -r BS NBS TRAIN_ITERS MOE_TYPE MODEL_SIZE CHECKPOINT CHECKPOINT_NUM_LAYERS PP_PARTITION PLANNER_MODE <<< "$batch_config"

            
            RUN_PLANNER="false"

            # 2. Check if Planner is enabled for this run
            if [[ "$PLANNER_MODE" == "yes-planner" ]]; then

                RUN_PLANNER="true"
                # A. Determine Profiling Model Name (Ensure _1L)
                PROF_MODEL_NAME="${MODEL_SIZE}_1L"
                EP_NODES=$(( EP_PARALLEL_SIZE / 8 ))

                # Call to get cache file 
                CACHE_FILENAME=$(python3 ../utils/model_registry.py get_filename \
                    --model_size "$MODEL_SIZE" \
                    --nodes "$EP_NODES" \
                    --total_gpus "$EP_PARALLEL_SIZE" \
                    --mbs 1)
                
                CACHE_FILE="./planner_profiling_cache/${CACHE_FILENAME}"

                # C. Check if Cache Exists
                if [ -f "$CACHE_FILE" ]; then
                    echo "  [Planner] Cache found: $CACHE_FILENAME"
                else
                    echo "  [Planner] Cache MISSING $CACHE_FILE for $PROF_MODEL_NAME. Continue to next one"

                    continue 
                fi

            fi

            if [[ "$CHECKPOINT" == "ckpt" ]]; then
                ACTIVATION_CHECKPOINT="true"
                DYNAMIC_CHECKPOINT="False" # Capital F
            elif [[ "$CHECKPOINT" == "dynamic-ckpt" ]]; then
                ACTIVATION_CHECKPOINT="true"
                DYNAMIC_CHECKPOINT="True" # Capital T
            else
                ACTIVATION_CHECKPOINT="false"
                DYNAMIC_CHECKPOINT="False" # Capital F
            fi

            if [[ "$PP_PARTITION" == "uneven" ]]; then
                UNEVEN_PP="True" # Capital T
            else
                UNEVEN_PP="False" # Capital F
            fi

            RUN_TYPE="pp"
            JOB_NAME="${RUN_TYPE}_n${NODES}g${TOTAL_GPUS}_ep${EP_PARALLEL_SIZE}_nbs${NBS}_i${TRAIN_ITERS}_${MOE_TYPE}"
            TEMP_SLURM_SCRIPT="temp_slurm_${JOB_NAME}.slurm"

            echo "  - Generating job: ${JOB_NAME}"

            sed -e "s/{{RUN_TYPE}}/${RUN_TYPE}/g" \
                -e "s/{{NODES}}/${NODES}/g" \
                -e "s/{{TOTAL_GPUS}}/${TOTAL_GPUS}/g" \
                -e "s/{{PARTITION}}/${PARTITION}/g" \
                -e "s/{{BATCH_SIZE}}/${BS}/g" \
                -e "s/{{NUM_BATCHES}}/${NBS}/g" \
                -e "s/{{PP_SIZE}}/${PP_SIZE}/g" \
                -e "s/{{EP_PARALLEL_SIZE}}/${EP_PARALLEL_SIZE}/g" \
                -e "s/{{MP_SIZE}}/${MP_SIZE}/g" \
                -e "s/{{TRAIN_ITERS}}/${TRAIN_ITERS}/g" \
                -e "s/{{MOE_TYPE}}/${MOE_TYPE}/g" \
                -e "s/{{MODEL_SIZE}}/${MODEL_SIZE}/g" \
                -e "s/{{ACTIVATION_CHECKPOINT}}/${ACTIVATION_CHECKPOINT}/g" \
                -e "s/{{CHECKPOINT_NUM_LAYERS}}/${CHECKPOINT_NUM_LAYERS}/g" \
                -e "s/{{DYNAMIC_CHECKPOINT}}/${DYNAMIC_CHECKPOINT}/g" \
                -e "s/{{UNEVEN_PP}}/${UNEVEN_PP}/g" \
                -e "s/{{RUN_PLANNER}}/${RUN_PLANNER}/g" \
                ${TEMPLATE_FILE} > ${TEMP_SLURM_SCRIPT}

            sbatch ${TEMP_SLURM_SCRIPT}
            # rm ${TEMP_SLURM_SCRIPT}
            sleep 1
        done
    done
done


# --- LAUNCH EXPERT PARALLEL (EP) RUNS ---
echo
echo "-----------------------------------------------"
echo "--- Submitting Expert Parallel (EP) jobs ---"
echo "-----------------------------------------------"

# Iterate over the node configurations defined in the EP batch map
for node_key in "${!EP_BATCH_MAP[@]}"; do
    IFS=':' read -r NODES EP_PARALLEL_SIZE MP_SIZE <<< "$node_key"

    # Get the specific batch configs for this node setup
    batch_configs_string=${EP_BATCH_MAP[$node_key]}

    # Set partition for the new environment
    PARTITION="batch"
    # MP_SIZE=2


    # For EP runs, PP=1 and EP=Total GPUs
    PP_SIZE=1
    TOTAL_GPUS=$(( NODES * 8 ))
    
    echo "Found EP configurations for ${NODES} nodes, ${TOTAL_GPUS} GPUs..."


    for batch_config in $batch_configs_string; do
        IFS=':' read -r BS NBS TRAIN_ITERS MOE_TYPE MODEL_SIZE CHECKPOINT CHECKPOINT_NUM_LAYERS <<< "$batch_config"

        if [[ "$CHECKPOINT" == "ckpt" ]]; then
            ACTIVATION_CHECKPOINT="true"
        else
            ACTIVATION_CHECKPOINT="false"
        fi

        RUN_TYPE="ep"
        JOB_NAME="${RUN_TYPE}_n${NODES}g${TOTAL_GPUS}_ep${EP_PARALLEL_SIZE}_nbs${NBS}_i${TRAIN_ITERS}_${MOE_TYPE}"
        TEMP_SLURM_SCRIPT="temp_slurm_${JOB_NAME}.slurm"

        echo "  - Generating job: ${JOB_NAME}"

        sed -e "s/{{RUN_TYPE}}/${RUN_TYPE}/g" \
            -e "s/{{NODES}}/${NODES}/g" \
            -e "s/{{TOTAL_GPUS}}/${TOTAL_GPUS}/g" \
            -e "s/{{PARTITION}}/${PARTITION}/g" \
            -e "s/{{BATCH_SIZE}}/${BS}/g" \
            -e "s/{{NUM_BATCHES}}/${NBS}/g" \
            -e "s/{{PP_SIZE}}/${PP_SIZE}/g" \
            -e "s/{{EP_PARALLEL_SIZE}}/${EP_PARALLEL_SIZE}/g" \
            -e "s/{{MP_SIZE}}/${MP_SIZE}/g" \
            -e "s/{{TRAIN_ITERS}}/${TRAIN_ITERS}/g" \
            -e "s/{{MOE_TYPE}}/${MOE_TYPE}/g" \
            -e "s/{{MODEL_SIZE}}/${MODEL_SIZE}/g" \
                -e "s/{{ACTIVATION_CHECKPOINT}}/${ACTIVATION_CHECKPOINT}/g" \
                -e "s/{{CHECKPOINT_NUM_LAYERS}}/${CHECKPOINT_NUM_LAYERS}/g" \
            ${TEMPLATE_FILE} > ${TEMP_SLURM_SCRIPT}

        sbatch ${TEMP_SLURM_SCRIPT}
        # rm ${TEMP_SLURM_SCRIPT}
        sleep 1
    done
done







# Iterate over the node configurations defined in the EP batch map
for node_key in "${!PROFILE_MAP[@]}"; do
    IFS=':' read -r NODES TOTAL_GPUS MP_SIZE <<< "$node_key"

    # Get the specific batch configs for this node setup
    batch_configs_string=${PROFILE_MAP[$node_key]}

    # Set partition for the new environment
    PARTITION="batch"
    # MP_SIZE=2
    TEMPLATE_PLANNER_FILE="planner.slurm.template"
    
    echo "Found EP configurations for ${NODES} nodes, ${TOTAL_GPUS} GPUs..."

    # For EP runs, PP=1 and EP=Total GPUs
    PP_SIZE=1
    EP_PARALLEL_SIZE=$(( TOTAL_GPUS / MP_SIZE ))

    for batch_config in $batch_configs_string; do
        IFS=':' read -r BS NBS TRAIN_ITERS MOE_TYPE MODEL_SIZE CHECKPOINT CHECKPOINT_NUM_LAYERS <<< "$batch_config"

        if [[ "$CHECKPOINT" == "ckpt" ]]; then
            ACTIVATION_CHECKPOINT="true"
        else
            ACTIVATION_CHECKPOINT="false"
        fi

        for BS in {1..10}; do

            RUN_TYPE="pr"
            JOB_NAME="${RUN_TYPE}_n${NODES}g${TOTAL_GPUS}_ep${EP_PARALLEL_SIZE}_nbs${NBS}_i${TRAIN_ITERS}_${MOE_TYPE}"
            TEMP_SLURM_SCRIPT="temp_slurm_${JOB_NAME}.slurm"

            echo "  - Generating job: ${JOB_NAME}, BS:$BS"

            sed -e "s/{{RUN_TYPE}}/${RUN_TYPE}/g" \
                -e "s/{{NODES}}/${NODES}/g" \
                -e "s/{{TOTAL_GPUS}}/${TOTAL_GPUS}/g" \
                -e "s/{{PARTITION}}/${PARTITION}/g" \
                -e "s/{{BATCH_SIZE}}/${BS}/g" \
                -e "s/{{NUM_BATCHES}}/${NBS}/g" \
                -e "s/{{PP_SIZE}}/${PP_SIZE}/g" \
                -e "s/{{EP_PARALLEL_SIZE}}/${EP_PARALLEL_SIZE}/g" \
                -e "s/{{MP_SIZE}}/${MP_SIZE}/g" \
                -e "s/{{TRAIN_ITERS}}/${TRAIN_ITERS}/g" \
                -e "s/{{MOE_TYPE}}/${MOE_TYPE}/g" \
                -e "s/{{MODEL_SIZE}}/${MODEL_SIZE}/g" \
                -e "s/{{ACTIVATION_CHECKPOINT}}/${ACTIVATION_CHECKPOINT}/g" \
                -e "s/{{CHECKPOINT_NUM_LAYERS}}/0/g" \
                -e "s/{{DYNAMIC_CHECKPOINT}}/False/g" \
                -e "s/{{UNEVEN_PP}}/False/g" \
                -e "s/{{PROFILING_ENABLED}}/true/g" \
                ${TEMPLATE_PLANNER_FILE} > ${TEMP_SLURM_SCRIPT}

            sbatch ${TEMP_SLURM_SCRIPT}
            # rm ${TEMP_SLURM_SCRIPT}
            sleep 1
        done 
    done
done



echo
echo "All jobs submitted."