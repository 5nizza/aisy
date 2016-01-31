from utils import find_files


RUN_TOOL = "./aisy.py -q"
CHECK_ANSWER = "/check_model.sh"
DEFAULT_OUTPUT_FOLDER = None   # 'None' means do not create a permanent folder


# TO BE REMOVED
############################################################################
# When used with --mc flag, requires `iimc` model checker (http://iimc.colorado.edu/).
# Change the paths according to your setup.
# Used only if run with --mc
CHECK_MODEL_RC_CORRECT = 0

############################################################################
EXIT_STATUS_REALIZABLE = 10
EXIT_STATUS_UNREALIZABLE = 20
