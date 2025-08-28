import torch
import librosa
import time
import json
import pandas as pd
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Wav2Vec2Processor, Wav2Vec2ForCTC
from datetime import datetime

# Check if MPS is available (for Apple Silicon)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Define your audio files
audio_files = [
    "audio1.wav",  # Replace with your actual file paths
    "audio2.wav"   # Replace with your actual file paths
]

# Model configurations
models_config = {
    "finetuned_whisper": {
        "model_id": "dmatekenya/whisper-large-v3-chichewa",
        "revision": "bff60fb08ba9f294e05bfcab4306f30b6a0cfc0a",
        "base_model": "openai/whisper-large-v3",
        "language": "shona",
        "type": "whisper"
    },
    "base_whisper": {
        "model_id": "openai/whisper-large-v3",
        "revision": None,
        "base_model": "openai/whisper-large-v3",
        "language": "shona",
        "type": "whisper"
    },
    "meta_mms": {
        "model_id": "facebook/mms-1b-all",  # You may need to adjust this
        "revision": None,
        "base_model": None,
        "language": "sn",  # Shona language code for MMS
        "type": "mms"
    }
}

def load_whisper_model(config):
    """Load Whisper model and processor"""
    print(f"Loading Whisper model: {config['model_id']}")
    
    processor = WhisperProcessor.from_pretrained(
        config['base_model'],
        language=config['language'],
        task="transcribe"
    )
    
    if config['revision']:
        model = WhisperForConditionalGeneration.from_pretrained(
            config['model_id'],
            revision=config['revision']
        )
    else:
        model = WhisperForConditionalGeneration.from_pretrained(config['model_id'])
    
    model = model.to(device)
    model.eval()
    
    return model, processor

def load_mms_model(config):
    """Load MMS model and processor"""
    print(f"Loading MMS model: {config['model_id']}")
    
    processor = Wav2Vec2Processor.from_pretrained(config['model_id'])
    model = Wav2Vec2ForCTC.from_pretrained(config['model_id'])
    
    model = model.to(device)
    model.eval()
    
    return model, processor

def transcribe_whisper(model, processor, audio_path):
    """Transcribe audio using Whisper model"""
    # Load audio
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Prepare input features
    input_features = processor(y, return_tensors="pt", sampling_rate=sr).input_features
    input_features = input_features.to(device)
    
    # Generate transcription
    start_time = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            inputs=input_features,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id
        )
    inference_time = time.time() - start_time
    
    # Decode transcription
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return transcription, inference_time

def transcribe_mms(model, processor, audio_path):
    """Transcribe audio using MMS model"""
    # Load audio
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Prepare input features
    input_values = processor(y, return_tensors="pt", sampling_rate=sr).input_values
    input_values = input_values.to(device)
    
    # Generate transcription
    start_time = time.time()
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
    inference_time = time.time() - start_time
    
    # Decode transcription
    transcription = processor.decode(predicted_ids[0])
    
    return transcription, inference_time

def get_audio_duration(audio_path):
    """Get audio duration in seconds"""
    y, sr = librosa.load(audio_path, sr=16000)
    return len(y) / sr

# Main comparison function
def run_comparison():
    results = []
    
    # Load all models
    models = {}
    
    # Load Whisper models
    for model_name in ["finetuned_whisper", "base_whisper"]:
        config = models_config[model_name]
        models[model_name] = load_whisper_model(config)
    
    # Load MMS model
    try:
        config = models_config["meta_mms"]
        models["meta_mms"] = load_mms_model(config)
        print("MMS model loaded successfully")
    except Exception as e:
        print(f"Failed to load MMS model: {e}")
        print("Continuing with Whisper models only...")
        models["meta_mms"] = None
    
    # Process each audio file
    for audio_file in audio_files:
        print(f"\n--- Processing {audio_file} ---")
        
        # Get audio duration
        try:
            duration = get_audio_duration(audio_file)
            print(f"Audio duration: {duration:.2f} seconds")
        except Exception as e:
            print(f"Could not load {audio_file}: {e}")
            continue
        
        # Test each model
        for model_name, model_data in models.items():
            if model_data is None:
                continue
                
            print(f"\nTesting {model_name}...")
            model, processor = model_data
            
            try:
                if model_name in ["finetuned_whisper", "base_whisper"]:
                    transcription, inference_time = transcribe_whisper(model, processor, audio_file)
                else:  # MMS
                    transcription, inference_time = transcribe_mms(model, processor, audio_file)
                
                # Calculate real-time factor (lower is better)
                rtf = inference_time / duration if duration > 0 else float('inf')
                
                result = {
                    "audio_file": audio_file,
                    "model_name": model_name,
                    "model_id": models_config[model_name]["model_id"],
                    "transcription": transcription,
                    "inference_time_seconds": round(inference_time, 3),
                    "audio_duration_seconds": round(duration, 2),
                    "real_time_factor": round(rtf, 3),
                    "timestamp": datetime.now().isoformat()
                }
                
                results.append(result)
                
                print(f"Inference time: {inference_time:.3f}s")
                print(f"Real-time factor: {rtf:.3f}x")
                print(f"Transcription preview: {transcription[:100]}...")
                
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                
                error_result = {
                    "audio_file": audio_file,
                    "model_name": model_name,
                    "model_id": models_config[model_name]["model_id"],
                    "transcription": f"ERROR: {str(e)}",
                    "inference_time_seconds": None,
                    "audio_duration_seconds": round(duration, 2) if 'duration' in locals() else None,
                    "real_time_factor": None,
                    "timestamp": datetime.now().isoformat()
                }
                results.append(error_result)
    
    return results

# Run the comparison
print("Starting model comparison...")
results = run_comparison()

# Save results to JSON
json_filename = f"transcription_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(json_filename, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Save results to CSV
csv_filename = f"transcription_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
df = pd.DataFrame(results)
df.to_csv(csv_filename, index=False, encoding='utf-8')

print(f"\n--- Results Summary ---")
print(f"Results saved to:")
print(f"- JSON: {json_filename}")
print(f"- CSV: {csv_filename}")

# Print summary table
if results:
    print(f"\nPerformance Summary:")
    summary_df = df[df['transcription'] != df['transcription'].str.startswith('ERROR', na=False)]
    if not summary_df.empty:
        summary = summary_df.groupby('model_name').agg({
            'inference_time_seconds': 'mean',
            'real_time_factor': 'mean'
        }).round(3)
        print(summary)
    
    print(f"\nSample transcriptions:")
    for result in results[:6]:  # Show first 6 results
        if not result['transcription'].startswith('ERROR'):
            print(f"\n{result['model_name']} on {result['audio_file']}:")
            print(f"{result['transcription'][:200]}...")
