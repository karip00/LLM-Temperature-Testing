import re
import time
import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import os
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, classification_report, confusion_matrix
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
import seaborn as sns
import matplotlib.pyplot as plt

logging.set_verbosity_error()

temper = 0.3

model_names = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Ministral-8B-Instruct-2410",
    "ibm-granite/granite-3.3-8b-instruct",
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "tiiuae/Falcon3-10B-Instruct",
    "upstage/SOLAR-10.7B-Instruct-v1.0",
    "deepseek-ai/deepseek-llm-7b-chat", 
    "Intel/neural-chat-7b-v3-3",
    "GSAI-ML/LLaDA-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "kakaocorp/kanana-1.5-8b-instruct-2505",
    "utter-project/EuroLLM-9B-Instruct",
    "Langboat/Mengzi3-8B-Chat",
    "zai-org/glm-4-9b-chat-hf",
]

from datasets import load_dataset

print("Loading SWAG dataset...")
test_dataset = load_dataset("swag", "regular", split="validation")
df = test_dataset.to_pandas()
df = df[:100]

df['context'] = df['sent1'] + ' ' + df['sent2']
df['choices'] = df.apply(lambda row: [row['ending0'], row['ending1'], row['ending2'], row['ending3']], axis=1)
df['label_text'] = df['label'].apply(lambda x: chr(65 + x))
df = df[['context', 'choices', 'label_text', 'label']].reset_index(drop=True)
print(f"Dataset loaded: {len(df)} samples")

MAX_INPUT_TOKENS = 200
MAX_NEW_TOKENS = 2
N_SAMPLES_PER_ITEM = 5
DEBUG_SAMPLES = 10

def get_gpu_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def get_gpu_memory_reserved():
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1024 / 1024
    return 0

def log_gpu_memory(stage, model_name):
    allocated = get_gpu_memory_usage()
    reserved = get_gpu_memory_reserved()
    print(f"[{stage}] {model_name} - GPU Memory: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")
    return allocated, reserved

def build_prompt(context, choices):
    return (
        "You are a commonsense reasoning assistant. "
        "Given a context, select the most plausible continuation from the choices. "
        "Respond ONLY with the letter (A, B, C, or D) of the correct choice.\n\n"
        f"Context: {context}\n\n"
        f"A) {choices[0]}\n"
        f"B) {choices[1]}\n"
        f"C) {choices[2]}\n"
        f"D) {choices[3]}\n\n"
        "Answer:"
    )

def extract_predicted_label(decoded_output):
    last_line = decoded_output.strip().split('\n')[-1]
    match = re.search(r"\b([ABCD])\b", last_line.upper())
    if match:
        return match.group(1)
    
    last_line_lower = last_line.lower()
    if 'choice a' in last_line_lower or 'option a' in last_line_lower:
        return 'A'
    elif 'choice b' in last_line_lower or 'option b' in last_line_lower:
        return 'B'
    elif 'choice c' in last_line_lower or 'option c' in last_line_lower:
        return 'C'
    elif 'choice d' in last_line_lower or 'option d' in last_line_lower:
        return 'D'
    
    return "unknown"

def log_evaluation_metrics(model_name, time_taken, vram_allocated, vram_reserved, 
                          log_file=f"./Metrics Reports/all_evaltime_{temper}_gpumem.txt"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    with open(log_file, "a") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"  - Total evaluation time: {time_taken:.2f} seconds\n")
        f.write(f"  - VRAM allocated by model: {vram_allocated:.2f} MB\n")
        f.write(f"  - VRAM reserved by model: {vram_reserved:.2f} MB\n")
        f.write(f"  - Date/Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 50 + "\n")
    
    print(f"Logged evaluation metrics for {model_name}:")
    print(f"   - Time: {time_taken:.2f} seconds")
    print(f"   - VRAM: {vram_allocated:.2f}MB allocated, {vram_reserved:.2f}MB reserved")

def load_model_and_tokenizer(model_id):
    try:
        model_name = model_id.split("/")[-1]
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        initial_allocated, initial_reserved = log_gpu_memory("Before model loading", model_name)
        
        print(f"Loading model: {model_id}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        print(f"Model loaded: {model_id}")
        
        final_allocated, final_reserved = log_gpu_memory("After model loading", model_name)
        
        model_vram_allocated = final_allocated - initial_allocated
        model_vram_reserved = final_reserved - initial_reserved
        
        print(f"Model VRAM Usage:")
        print(f"   - Allocated: {model_vram_allocated:.2f}MB")
        print(f"   - Reserved: {model_vram_reserved:.2f}MB")
        print(f"   - Total GPU Memory: {final_allocated:.2f}MB allocated, {final_reserved:.2f}MB reserved")
        
        return model, tokenizer, model_vram_allocated, model_vram_reserved
        
    except Exception as e:
        print(f"Failed to load model {model_id}: {str(e)}")
        return None, None, 0, 0

def evaluate_model_with_logging(model, tokenizer, df, model_name="model", 
                               n_samples_per_item=N_SAMPLES_PER_ITEM, debug_samples=DEBUG_SAMPLES,
                               save_path="./Results/test.csv", vram_allocated=0, vram_reserved=0):
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if os.path.exists(save_path):
        print(f"Results file already exists at {save_path}. Loading existing results.")
        return pd.read_csv(save_path)
    
    df_eval = df.copy()
    
    model.eval()
    
    if 'llm_label' not in df_eval.columns:
        df_eval['llm_label'] = None
    if 'self_consistency_votes' not in df_eval.columns:
        df_eval['self_consistency_votes'] = None
    
    print(f"Starting evaluation for {model_name}")
    print(f"Dataset size: {len(df_eval)} samples")
    print(f"Self-consistency samples per item: {n_samples_per_item}")
    
    start_time_evaluation = time.time()
    
    progress_bar = tqdm(df_eval.iterrows(), 
                       total=len(df_eval), 
                       desc=f"Evaluating {model_name}", 
                       dynamic_ncols=True)
    
    for idx, row in progress_bar:
        
        if pd.notna(row['llm_label']):
            continue
        
        context = row['context']
        choices = row['choices']
        prompt = build_prompt(context, choices)
        inputs = tokenizer(prompt, 
                          return_tensors="pt", 
                          truncation=True, 
                          max_length=MAX_INPUT_TOKENS).to("cuda")
        
        consistency_votes = []
        
        for sample_idx in range(n_samples_per_item):
            try:
                cache_flag = True
                if model_name == "LLaDA-8B-Instruct":
                    cache_flag = False
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        temperature=temper,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=cache_flag
                    )
                
                decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                predicted_label = extract_predicted_label(decoded_output)
                consistency_votes.append(predicted_label)
                
            except Exception as e:
                print(f"Error during generation for sample {idx}, vote {sample_idx}: {e}")
                consistency_votes.append("unknown")
        
        if consistency_votes:
            final_prediction = max(set(consistency_votes), key=consistency_votes.count)
        else:
            final_prediction = "unknown"
        
        df_eval.at[idx, 'llm_label'] = final_prediction
        df_eval.at[idx, 'self_consistency_votes'] = str(consistency_votes)
        
        if debug_samples > 0 and idx < debug_samples:
            print(f"[Sample {idx}] Final prediction: {final_prediction} | "
                  f"Consistency votes: {consistency_votes}")
            print(f"  Context: {context}")
            print(f"  Choices: {choices}")
            print(f"  Ground truth: {row['label_text']}")
            print("-" * 50)
    
    total_evaluation_time = time.time() - start_time_evaluation
    print(f"Evaluation complete for {model_name}")
    print(f"Total evaluation time: {total_evaluation_time:.2f} seconds")
    
    log_evaluation_metrics(model_name, total_evaluation_time, vram_allocated, vram_reserved)
    
    df_eval.to_csv(save_path, index=False)
    print(f"Results saved to: {save_path}")
    
    return df_eval

def calculate_basic_metrics(df_results):
    
    valid_mask = df_results['llm_label'] != 'unknown'
    valid_df = df_results[valid_mask].copy()
    
    if len(valid_df) == 0:
        print("No valid predictions found!")
        return {}
    
    y_true = valid_df['label_text']
    y_pred = valid_df['llm_label']
    
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_true, y_pred)
    
    labels = ['A', 'B', 'C', 'D']
    f1_scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0.0)
    
    total_samples = len(df_results)
    valid_samples = len(valid_df)
    invalid_samples = total_samples - valid_samples
    
    metrics = {
        "total_samples": total_samples,
        "valid_predictions": valid_samples,
        "invalid_predictions": invalid_samples,
        "valid_prediction_rate": valid_samples / total_samples,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "f1_a": f1_scores[0] if len(f1_scores) > 0 else 0.0,
        "f1_b": f1_scores[1] if len(f1_scores) > 1 else 0.0,
        "f1_c": f1_scores[2] if len(f1_scores) > 2 else 0.0,
        "f1_d": f1_scores[3] if len(f1_scores) > 3 else 0.0,
        "cohen_kappa": kappa
    }
    
    return metrics

def calculate_uncertainty_metrics(df_results):
    
    valid_mask = df_results['llm_label'] != 'unknown'
    valid_df = df_results[valid_mask].copy()
    
    if len(valid_df) == 0:
        print("No valid predictions for uncertainty calculation!")
        return {}
    
    sc_entropy_list = []
    js_divergence_list = []
    confidence_scores = []
    
    for idx, row in valid_df.iterrows():
        
        votes_str = row['self_consistency_votes']
        try:
            votes_list = eval(votes_str)
        except:
            continue
        
        vote_counts = pd.Series(votes_list).value_counts(normalize=True)
        
        sc_entropy = entropy(vote_counts, base=2)
        sc_entropy_list.append(sc_entropy)
        
        max_vote_proportion = vote_counts.max()
        confidence_scores.append(max_vote_proportion)
        
        true_label = row['label_text']
        if true_label == 'A':
            ground_truth_dist = np.array([1.0, 0.0, 0.0, 0.0])
        elif true_label == 'B':
            ground_truth_dist = np.array([0.0, 1.0, 0.0, 0.0])
        elif true_label == 'C':
            ground_truth_dist = np.array([0.0, 0.0, 1.0, 0.0])
        else:
            ground_truth_dist = np.array([0.0, 0.0, 0.0, 1.0])
        
        pred_dist = np.array([
            vote_counts.get('A', 0.0),
            vote_counts.get('B', 0.0),
            vote_counts.get('C', 0.0),
            vote_counts.get('D', 0.0)
        ])
        
        if pred_dist.sum() > 0:
            pred_dist = pred_dist / pred_dist.sum()
        
        js_div = jensenshannon(pred_dist, ground_truth_dist)
        js_divergence_list.append(js_div)
    
    uncertainty_metrics = {
        "mean_entropy": np.mean(sc_entropy_list) if sc_entropy_list else 0,
        "std_entropy": np.std(sc_entropy_list) if sc_entropy_list else 0,
        "mean_confidence": np.mean(confidence_scores) if confidence_scores else 0,
        "std_confidence": np.std(confidence_scores) if confidence_scores else 0,
        "mean_js_divergence": np.mean(js_divergence_list) if js_divergence_list else 0,
        "std_js_divergence": np.std(js_divergence_list) if js_divergence_list else 0,
        "low_confidence_samples": sum(1 for conf in confidence_scores if conf < 0.6),
        "high_uncertainty_samples": sum(1 for ent in sc_entropy_list if ent > 0.5),
        "perfect_agreement_samples": sum(1 for conf in confidence_scores if conf == 1.0)
    }
    
    return uncertainty_metrics

def analyze_swag_patterns(df_results):
    
    valid_mask = df_results['llm_label'] != 'unknown'
    valid_df = df_results[valid_mask].copy()
    
    if len(valid_df) == 0:
        return {}
    
    true_dist = valid_df['label_text'].value_counts()
    pred_dist = valid_df['llm_label'].value_counts()
    
    y_true = valid_df['label_text']
    y_pred = valid_df['llm_label']
    
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=['A', 'B', 'C', 'D'], zero_division=0.0
    )
    
    swag_metrics = {
        "true_a_count": int(true_dist.get('A', 0)),
        "true_b_count": int(true_dist.get('B', 0)),
        "true_c_count": int(true_dist.get('C', 0)),
        "true_d_count": int(true_dist.get('D', 0)),
        "pred_a_count": int(pred_dist.get('A', 0)),
        "pred_b_count": int(pred_dist.get('B', 0)),
        "pred_c_count": int(pred_dist.get('C', 0)),
        "pred_d_count": int(pred_dist.get('D', 0)),
        "a_precision": precision[0],
        "b_precision": precision[1],
        "c_precision": precision[2],
        "d_precision": precision[3],
        "a_recall": recall[0],
        "b_recall": recall[1],
        "c_recall": recall[2],
        "d_recall": recall[3],
        "a_f1": f1[0],
        "b_f1": f1[1],
        "c_f1": f1[2],
        "d_f1": f1[3],
    }
    
    return swag_metrics

def generate_evaluation_report(df_results, model_name=None, save_report=True):
    
    print("SWAG COMMONSENSE REASONING - EVALUATION REPORT")
    print("=" * 60)
    
    if model_name:
        print(f"Model: {model_name}")
        print("-" * 60)
    
    print("\nCLASSIFICATION PERFORMANCE:")
    basic_metrics = calculate_basic_metrics(df_results)
    
    if basic_metrics:
        print(f"Total Samples: {basic_metrics['total_samples']}")
        print(f"Valid Predictions: {basic_metrics['valid_predictions']}")
        print(f"Invalid Predictions: {basic_metrics['invalid_predictions']}")
        print(f"Valid Prediction Rate: {basic_metrics['valid_prediction_rate']:.3f}")
        print(f"Accuracy: {basic_metrics['accuracy']:.3f}")
        print(f"Macro F1: {basic_metrics['macro_f1']:.3f}")
        print(f"Weighted F1: {basic_metrics['weighted_f1']:.3f}")
        print(f"F1 (A): {basic_metrics['f1_a']:.3f}")
        print(f"F1 (B): {basic_metrics['f1_b']:.3f}")
        print(f"F1 (C): {basic_metrics['f1_c']:.3f}")
        print(f"F1 (D): {basic_metrics['f1_d']:.3f}")
        print(f"Cohen's Kappa: {basic_metrics['cohen_kappa']:.3f}")
    
    print("\nCOMMONSENSE REASONING ANALYSIS:")
    swag_metrics = analyze_swag_patterns(df_results)
    
    if swag_metrics:
        print(f"True A: {swag_metrics['true_a_count']}")
        print(f"True B: {swag_metrics['true_b_count']}")
        print(f"True C: {swag_metrics['true_c_count']}")
        print(f"True D: {swag_metrics['true_d_count']}")
        print(f"Predicted A: {swag_metrics['pred_a_count']}")
        print(f"Predicted B: {swag_metrics['pred_b_count']}")
        print(f"Predicted C: {swag_metrics['pred_c_count']}")
        print(f"Predicted D: {swag_metrics['pred_d_count']}")
        print(f"\nPrecision - A: {swag_metrics['a_precision']:.3f}, "
              f"B: {swag_metrics['b_precision']:.3f}, "
              f"C: {swag_metrics['c_precision']:.3f}, "
              f"D: {swag_metrics['d_precision']:.3f}")
        print(f"Recall - A: {swag_metrics['a_recall']:.3f}, "
              f"B: {swag_metrics['b_recall']:.3f}, "
              f"C: {swag_metrics['c_recall']:.3f}, "
              f"D: {swag_metrics['d_recall']:.3f}")
    
    print("\nUNCERTAINTY & CONSISTENCY:")
    uncertainty_metrics = calculate_uncertainty_metrics(df_results)
    
    if uncertainty_metrics:
        print(f"Mean Entropy: {uncertainty_metrics['mean_entropy']:.3f} (±{uncertainty_metrics['std_entropy']:.3f})")
        print(f"Mean Confidence: {uncertainty_metrics['mean_confidence']:.3f} (±{uncertainty_metrics['std_confidence']:.3f})")
        print(f"Mean JS Divergence: {uncertainty_metrics['mean_js_divergence']:.3f} (±{uncertainty_metrics['std_js_divergence']:.3f})")
        print(f"Low Confidence Samples (<0.6): {uncertainty_metrics['low_confidence_samples']}")
        print(f"High Uncertainty Samples (>0.5 entropy): {uncertainty_metrics['high_uncertainty_samples']}")
        print(f"Perfect Agreement Samples: {uncertainty_metrics['perfect_agreement_samples']}")
    
    valid_mask = df_results['llm_label'] != 'unknown'
    if valid_mask.sum() > 0:
        valid_df = df_results[valid_mask]
        y_true = valid_df['label_text']
        y_pred = valid_df['llm_label']
        
        print("\nDETAILED CLASSIFICATION REPORT:")
        print(classification_report(y_true, y_pred))
        
        print("\nCONFUSION MATRIX:")
        cm = confusion_matrix(y_true, y_pred, labels=['A', 'B', 'C', 'D'])
        print("Predicted:")
        print(f"         A    B    C    D")
        print(f"True A  {cm[0,0]:3d}  {cm[0,1]:3d}  {cm[0,2]:3d}  {cm[0,3]:3d}")
        print(f"    B   {cm[1,0]:3d}  {cm[1,1]:3d}  {cm[1,2]:3d}  {cm[1,3]:3d}")
        print(f"    C   {cm[2,0]:3d}  {cm[2,1]:3d}  {cm[2,2]:3d}  {cm[2,3]:3d}")
        print(f"    D   {cm[3,0]:3d}  {cm[3,1]:3d}  {cm[3,2]:3d}  {cm[3,3]:3d}")
    
    all_metrics = {
        **basic_metrics,
        **swag_metrics,
        **uncertainty_metrics,
        "model_name": model_name,
        "task": "swag_commonsense_reasoning"
    }
    
    if save_report and model_name:
        report_path = f"./Metrics Reports/{model_name}_{temper}_swag_metrics_report.txt"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, "w") as f:
            f.write(f"SWAG COMMONSENSE REASONING EVALUATION REPORT FOR {model_name}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("CLASSIFICATION PERFORMANCE:\n")
            for key, value in basic_metrics.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\nCOMMONSENSE REASONING ANALYSIS:\n")
            for key, value in swag_metrics.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\nUNCERTAINTY & CONSISTENCY:\n")
            for key, value in uncertainty_metrics.items():
                f.write(f"{key}: {value}\n")

            if valid_mask.sum() > 0:
                f.write("\nDETAILED CLASSIFICATION REPORT:\n")
                f.write(classification_report(y_true, y_pred))
                
                f.write("\nCONFUSION MATRIX:\n")
                f.write("Predicted:\n")
                f.write(f"         A    B    C    D\n")
                f.write(f"True A  {cm[0,0]:3d}  {cm[0,1]:3d}  {cm[0,2]:3d}  {cm[0,3]:3d}\n")
                f.write(f"    B   {cm[1,0]:3d}  {cm[1,1]:3d}  {cm[1,2]:3d}  {cm[1,3]:3d}\n")
                f.write(f"    C   {cm[2,0]:3d}  {cm[2,1]:3d}  {cm[2,2]:3d}  {cm[2,3]:3d}\n")
                f.write(f"    D   {cm[3,0]:3d}  {cm[3,1]:3d}  {cm[3,2]:3d}  {cm[3,3]:3d}\n")
        
        print(f"\nDetailed report saved to: {report_path}")
    
    return all_metrics

def main():
    
    print("STARTING MULTI-MODEL SWAG EVALUATION PIPELINE")
    print("=" * 70)
    print(f"Models to evaluate: {len(model_names)}")
    for i, model_name in enumerate(model_names, 1):
        print(f"   {i}. {model_name}")
    print("=" * 70)
    
    os.makedirs("./LLM Labels", exist_ok=True)
    os.makedirs("./Metrics Reports", exist_ok=True)
    
    successful_evaluations = 0
    failed_evaluations = 0
    
    for model_idx, model_id in enumerate(model_names, 1):
        
        print(f"\nPROCESSING MODEL {model_idx}/{len(model_names)}: {model_id}")
        print("=" * 70)
        
        try:
            model_name = model_id.split("/")[-1]
            save_path = Path(f"./LLM Labels/{model_name}_{temper}_swag.csv")
            
            if save_path.exists():
                print(f"Results already exist for {model_name}. Loading and evaluating metrics only.")
                df_results = pd.read_csv(save_path)
                
                print(f"\nGenerating evaluation report for {model_name}...")
                final_metrics = generate_evaluation_report(df_results, model_name=model_name)
                successful_evaluations += 1
                continue
            
            print(f"\nSTEP 1: Loading {model_name}...")
            model, tokenizer, model_vram_allocated, model_vram_reserved = load_model_and_tokenizer(model_id)
            
            if model is None or tokenizer is None:
                print(f"Failed to load {model_name}. Skipping...")
                failed_evaluations += 1
                continue
            
            print(f"\nSTEP 2: Generating predictions for {model_name}...")
            df_results = evaluate_model_with_logging(
                model=model,
                tokenizer=tokenizer,
                df=df.copy(),
                model_name=model_name,
                n_samples_per_item=N_SAMPLES_PER_ITEM,
                debug_samples=DEBUG_SAMPLES,
                save_path=save_path,
                vram_allocated=model_vram_allocated,
                vram_reserved=model_vram_reserved
            )
            
            print(f"\nSTEP 3: Calculating evaluation metrics for {model_name}...")
            final_metrics = generate_evaluation_report(df_results, model_name=model_name)
            
            print(f"\nSTEP 4: Cleaning up GPU memory for {model_name}...")
            
            del model
            del tokenizer
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            cleanup_allocated, cleanup_reserved = log_gpu_memory("After cleanup", model_name)
            
            print(f"Successfully completed evaluation for {model_name}")
            successful_evaluations += 1
            
        except Exception as e:
            print(f"Error processing {model_id}: {str(e)}")
            print(f"Attempting cleanup and continuing to next model...")
            
            try:
                if 'model' in locals():
                    del model
                if 'tokenizer' in locals():
                    del tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            
            failed_evaluations += 1
            continue
    
    print("\n" + "=" * 70)
    print("MULTI-MODEL EVALUATION PIPELINE COMPLETED!")
    print("=" * 70)
    print(f"Successful evaluations: {successful_evaluations}")
    print(f"Failed evaluations: {failed_evaluations}")
    print(f"Total models processed: {len(model_names)}")
    
    if successful_evaluations > 0:
        print(f"\nResults saved in:")
        print(f"   - Predictions: ./LLM Labels/")
        print(f"   - Metrics reports: ./Metrics Reports/")
        print(f"   - Time & VRAM logs: ./Metrics Reports/all_evaltime_gpumem_{temper}.txt")
    
    print(f"\nPipeline execution completed!")

if __name__ == "__main__":
    main()