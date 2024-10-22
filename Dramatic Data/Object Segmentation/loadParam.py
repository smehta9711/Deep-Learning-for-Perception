import os

HOME_PATH   =   os.path.expanduser("~")
JOB_ID      =   "run5"
MODEL_NAME  =   "WindowSeg"

DS_PATH     =   "/home/sviswasam/sf/sf/dataset/"   # change
OUT_PATH    =   "/home/sviswasam/sf/sf/new_UNet/output"

JOB_FOLDER  =   os.path.join(OUT_PATH, JOB_ID)
TRAINED_MDL_PATH    =   os.path.join(JOB_FOLDER, "parameters")
BATCH_SIZE          =   64
LR                  =   1e-4
LOG_BATCH_INTERVAL  =   1
LOG_WANDB = True
NUM_WORKERS  =   8