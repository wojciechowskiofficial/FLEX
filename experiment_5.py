from male import run_pipeline_single_decision
import os
from readability import Readability


EXPLANATIONS_PARENT_PATH = "./various_method_explanations"
RESULTS_FILE_PARENT = "./experiment_5_results_v2"


def Main():
    male_out, sat_out, nlx_out = "", "", ""
    
    for explanation_path in os.listdir(EXPLANATIONS_PARENT_PATH):
        with open(os.path.join(EXPLANATIONS_PARENT_PATH, explanation_path, explanation_path + ".JPEG_male_soft_gpt4_explanation.txt"), "r") as f:
            explanation = f.readline()
        male_out += f"{explanation} "
    r = Readability(male_out)
    with open(os.path.join(RESULTS_FILE_PARENT, "experiment_5_results_male.csv"), "w") as f:
        write_results(f, r)
        
    for explanation_path in os.listdir(EXPLANATIONS_PARENT_PATH):
        with open(os.path.join(EXPLANATIONS_PARENT_PATH, explanation_path,  explanation_path + ".JPEG_nlx-gpt_explanation.txt"), "r") as f:
            explanation = f.readline()
        nlx_out += f"{explanation} "
    r = Readability(nlx_out)
    with open(os.path.join(RESULTS_FILE_PARENT, "experiment_5_results_nlx.csv"), "w") as f:
        write_results(f, r)
        
    for explanation_path in os.listdir(EXPLANATIONS_PARENT_PATH):
        with open(os.path.join(EXPLANATIONS_PARENT_PATH, explanation_path,  explanation_path + ".JPEG_showattendtell_explanation.txt"), "r") as f:
            explanation = f.readline()
        sat_out += f"{explanation} "
    r = Readability(sat_out)
    with open(os.path.join(RESULTS_FILE_PARENT, "experiment_5_results_sat.csv"), "w") as f:
        write_results(f, r)


def write_results(f, r):
    f.write("name,score,grade-level\n")
    f.write(f"flesch-kincaid,{r.flesch_kincaid().score},{r.flesch_kincaid().grade_level}\n")
    f.write(f"flesch,{r.flesch().score},{r.flesch().grade_levels}\n")
    f.write(f"gunning-fog,{r.gunning_fog().score},{r.gunning_fog().grade_level}\n")
    f.write(f"coleman_liau,{r.coleman_liau().score},{r.coleman_liau().grade_level}\n")
    f.write(f"dalle-chall,{r.dale_chall().score},{r.dale_chall().grade_levels}\n")
    f.write(f"ARI,{r.ari().score},{r.ari().grade_levels}\n")
    f.write(f"linesear-write,{r.linsear_write().score},{r.linsear_write().grade_level}\n")
    f.write(f"smog,{r.smog().score},{r.smog().grade_level}\n")
    f.write(f"spache,{r.spache().score},{r.spache().grade_level}\n")
            
    

if __name__ == '__main__':
    Main()