import os
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from sacrebleu import sentence_bleu
from evaluate import load
import numpy as np


def Main(args):    
    bertscore = load("bertscore")
    meteorscore = load('meteor')

    df = pd.read_csv(args.val_images_names_full_path, header=None)
    image_names = df.iloc[:,0].tolist()
    
    # Inter
    for noise_intensity in [.05, .2]:
        variation = f"noise_int_{noise_intensity}"
        bleu_list, bert_p_list, bert_r_list, bert_f1_list, meteor_list = [], [], [], [], []
        # Explain
        for image_name in tqdm(image_names):
            with open(os.path.join(args.output_parent_path, args.dataset, args.cnn, image_name[:-5], variation, "soft_explanation.txt"), "r") as f:
                noise_exp = f.read().strip()
            
            with open(os.path.join("basic_explanations_gpt4", args.dataset, args.cnn, image_name[:-5], "soft_explanation.txt"), "r") as f:
                og_exp = f.read().strip()
            
            bleu_list.append(sentence_bleu(hypothesis=noise_exp, references=[og_exp]).score)
            bert_dict = bertscore.compute(predictions=[noise_exp], references=[og_exp], lang="en")
            bert_p_list.append(bert_dict["precision"])
            bert_r_list.append(bert_dict["recall"])
            bert_f1_list.append(bert_dict["f1"])
            meteor_list.append(meteorscore.compute(predictions=[noise_exp], references=[og_exp], lang="en"))
            
        with open(f"4_1_scores_inter_{variation}.csv", "w") as f:
            f.write("metric,mean,std")
            f.write("\n")
            f.write(f"BLEU,{np.mean(bleu_list)},{np.std(bleu_list)}")
            f.write("\n")
            f.write(f"BERT_P,{np.mean(bert_p_list)},{np.std(bert_p_list)}")
            f.write("\n")
            f.write(f"BERT_R,{np.mean(bert_r_list)},{np.std(bert_r_list)}")
            f.write("\n")
            f.write(f"BERT_F1,{np.mean(bert_f1_list)},{np.std(bert_f1_list)}")
            f.write("\n")
            f.write(f"METEOR,{np.mean(meteor_list)},{np.std(meteor_list)}")

    
    # Intra
    for noise_intensity in [.05, .2]:
        variation = f"noise_int_{noise_intensity}"
        bleu_list, bert_p_list, bert_r_list, bert_f1_list, meteor_list = [], [], [], [], []
        # Explain
        for first, second in zip(image_names[::2], image_names[1::2]):
            with open(os.path.join(args.output_parent_path, args.dataset, args.cnn, first[:-5], variation, "soft_explanation.txt"), "r") as f:
                noise_exp_first = f.read().strip()
                
            with open(os.path.join(args.output_parent_path, args.dataset, args.cnn, second[:-5], variation, "soft_explanation.txt"), "r") as f:
                noise_exp_second = f.read().strip()
            
            bleu_list.append(sentence_bleu(hypothesis=noise_exp_first, references=[noise_exp_second]).score)
            bert_dict = bertscore.compute(predictions=[noise_exp_first], references=[noise_exp_second], lang="en")
            bert_p_list.append(bert_dict["precision"])
            bert_r_list.append(bert_dict["recall"])
            bert_f1_list.append(bert_dict["f1"])
            meteor_list.append(meteorscore.compute(predictions=[noise_exp_first], references=[noise_exp_second], lang="en"))
            
            # reverse
            bleu_list.append(sentence_bleu(hypothesis=noise_exp_second, references=[noise_exp_first]).score)
            bert_dict = bertscore.compute(predictions=[noise_exp_second], references=[noise_exp_first], lang="en")
            bert_p_list.append(bert_dict["precision"])
            bert_r_list.append(bert_dict["recall"])
            bert_f1_list.append(bert_dict["f1"])
            meteor_list.append(meteorscore.compute(predictions=[noise_exp_second], references=[noise_exp_first], lang="en"))
            
        with open(f"4_1_scores_intra_{variation}.csv", "w") as f:
            f.write("metric,mean,std")
            f.write("\n")
            f.write(f"BLEU,{np.mean(bleu_list)},{np.std(bleu_list)}")
            f.write("\n")
            f.write(f"BERT_P,{np.mean(bert_p_list)},{np.std(bert_p_list)}")
            f.write("\n")
            f.write(f"BERT_R,{np.mean(bert_r_list)},{np.std(bert_r_list)}")
            f.write("\n")
            f.write(f"BERT_F1,{np.mean(bert_f1_list)},{np.std(bert_f1_list)}")
            f.write("\n")
            f.write(f"METEOR,{np.mean(meteor_list)},{np.std(meteor_list)}")



if __name__ == "__main__":
    parser = ArgumentParser(description='Run pipeline on whole validation set')
    parser.add_argument("--val_set_parent_path", default="./val")
    parser.add_argument("--val_images_names_full_path", default="./metadata/val_paths_experiments_0.csv")
    parser.add_argument("--milan_descriptions_full_path", default="./milan_results/resnet18_imagenet.csv")
    parser.add_argument("--openai_api_token_full_path", default="/home/adamwsl/.gpt_api_token/token.txt")
    parser.add_argument("--cnn", default="resnet18")
    parser.add_argument("--dataset", default="imagenet")
    parser.add_argument("--output_parent_path", default="./experiment_4_1_results")
    parser.add_argument("--explanation_type", default="soft")
    parser.add_argument("--do_not_copy_image", action="store_true", default=False)
    parser.add_argument("--dataset_class_names", default="./metadata/imagenet_classes.txt")
    args = parser.parse_args()
    
    Main(args)