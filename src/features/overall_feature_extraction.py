import yaml
import torch
import librosa
import pandas as pd
import numpy as np
import os 
import random
import opensmile
from pathlib import Path
from tqdm import tqdm
import argparse
from transformers import Wav2Vec2Model, Wav2Vec2Processor, HubertModel, Wav2Vec2FeatureExtractor

def set_seed(seed):
    """Set random seed."""
    if seed == -1:
        seed = random.randint(0, 1000)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def read_huggingface_token(token_path):
    with open(token_path, 'r') as f:
        return f.read().strip()

class FeatureExtractionPipeline:
    def __init__(self, config, huggingface_token=None):
        self.config = config
        self.device = torch.device(self.config['models']['device'] if torch.cuda.is_available() else "cpu")
        self.target_sr = self.config['audio']['target_sr']
        self.huggingface_token = huggingface_token

   
        if self.config['extractors']['run_egmaps']:
            self.smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
            self.egmaps_p_keys = self.config['egmaps_settings']['prosody_keywords']
            self.egmaps_v_keys = self.config['egmaps_settings']['vq_keywords']

  
        if self.config['extractors']['run_wav2vec2']:
            m_id = self.config['models']['wav2vec2']
            self.w2v2_proc = Wav2Vec2Processor.from_pretrained(m_id, token=self.huggingface_token)
            self.w2v2_model = Wav2Vec2Model.from_pretrained(m_id, token=self.huggingface_token).to(self.device)
            self.w2v2_model.eval()

       
        if self.config['extractors']['run_hubert']:
            m_id = self.config['models']['hubert']
            self.hub_proc = Wav2Vec2FeatureExtractor.from_pretrained(m_id, token=self.huggingface_token)
            self.hub_model = HubertModel.from_pretrained(m_id, token=self.huggingface_token).to(self.device)
            self.hub_model.eval()


    def _get_ssl_features(self, audio_signal, processor, model):
        """Internal helper to remove repetition between HuBERT and Wav2Vec2."""
        inputs = processor(audio_signal, sampling_rate=self.target_sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}


        with torch.no_grad():
            outputs = model(**inputs)
            mean_pooled = torch.mean(outputs.last_hidden_state, dim=1) 
            
        return mean_pooled.cpu().numpy().flatten()

    def get_global_wav2vec2(self, audio_signal):
        return self._get_ssl_features(audio_signal, self.w2v2_proc, self.w2v2_model)

    def get_global_hubert(self, audio_signal):
        return self._get_ssl_features(audio_signal, self.hub_proc, self.hub_model)
    

    def get_global_egmaps(self, audio_prosody, audio_vq, file_id):
        df_p = self.smile.process_signal(audio_prosody, self.target_sr)
        df_v = self.smile.process_signal(audio_vq, self.target_sr)

        p_cols = [c for c in df_p.columns if any(k in c for k in self.egmaps_p_keys)]
        v_cols = [c for c in df_v.columns if any(k in c for k in self.egmaps_v_keys)]

        g_p = df_p[p_cols].reset_index(drop=True).add_prefix("prosody_")
        g_v = df_v[v_cols].reset_index(drop=True).add_prefix("vq_")

        combined = pd.concat([g_p, g_v], axis=1)
        combined.insert(0, "file_id", file_id)
        g_p.insert(0, "file_id", file_id)
        g_v.insert(0, "file_id", file_id)
        return g_p, g_v, combined
    
    def process_item(self, file_id, p_file, c_file, output_dir):
        """Processes a single audio pair and saves results."""
        sig_p, _ = librosa.load(p_file, sr=self.target_sr)
        sig_c, _ = librosa.load(c_file, sr=self.target_sr)
        
  
        if self.config['extractors']['run_egmaps']:
            if (output_dir / f"{file_id}_egmaps.parquet").exists():
                return  # Skip if already done
            g_p, g_v, combined_df = self.get_global_egmaps(sig_p, sig_c, file_id)
            g_p.to_parquet(output_dir / f"{file_id}_egmaps.parquet")
            g_v.to_parquet(output_dir / f"{file_id}_vq_egmaps.parquet")
            combined_df.to_parquet(output_dir / f"{file_id}_egmaps_all.parquet")


        if self.config['extractors']['run_wav2vec2']:
            if (output_dir / f"{file_id}_wav2vec2.parquet").exists():
                return  # Skip if already done
            w2v2_vec = self._get_ssl_features(sig_p, self.w2v2_proc, self.w2v2_model)
            pd.DataFrame(w2v2_vec.reshape(1, -1)).to_parquet(output_dir / f"{file_id}_w2v2.parquet")

        if self.config['extractors']['run_hubert']:
            if (output_dir / f"{file_id}_hubert.parquet").exists():
                return  # Skip if already done
            hub_vec = self._get_ssl_features(sig_p, self.hub_proc, self.hub_model)
            pd.DataFrame(hub_vec.reshape(1, -1)).to_parquet(output_dir / f"{file_id}_hubert.parquet")

        

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to raw audio")
    parser.add_argument("--output", type=str, required=True, help="Base output directory")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, required=True)
    parser.add_argument("--token_path", type=str, required=True)
    parser.add_argument("--task_id", type=int, default=None)
    parser.add_argument("--total_tasks", type=int, default=None)   
    args = parser.parse_args()


    task_id = args.task_id if args.task_id is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    total_tasks = args.total_tasks if args.total_tasks is not None else int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    with open(args.token_path, 'r') as f: hf_token = f.read().strip()
    with open (args.config, 'r') as f:
        config = yaml.safe_load(f)

    effective_seed = config.get('seed', 42) 
    set_seed(effective_seed)
    print(f"Using seed: {effective_seed}")

    out_dir = Path(args.output) / args.dataset_type / "shards"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_prosody_files = sorted(list(Path(args.input).glob("*_prosody.wav")))
    n = len(all_prosody_files)
    
    indices = range(task_id, n, total_tasks) 
    my_files = [all_prosody_files[i] for i in indices]
    
    print(f"Task {task_id}/{total_tasks}: Processing {len(my_files)} files.")
    
    pipeline = FeatureExtractionPipeline(config, huggingface_token=hf_token)


    for p_file in my_files:
        file_id = p_file.name.replace("_prosody.wav", "")
        c_file = p_file.parent / f"{file_id}_concat.wav"

        
        if not c_file.exists():
            print(f"Concatenated file missing for {file_id}, skipping.")
            continue

        try:
            pipeline.process_item(file_id, p_file, c_file, out_dir)
            print(f"Processed {file_id}")
        except Exception as e:
            print(f"Error processing {file_id}: {e}")

if __name__ == "__main__":
    main()