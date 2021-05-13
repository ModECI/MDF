"""Create the MDF examples."""
import os
from modeci_mdf.interfaces.actr import actr_to_mdf

if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    for f in os.listdir(curr_dir):
        if f.endswith(".lisp"):
            actr_to_mdf(os.path.realpath(curr_dir + "/" + f))