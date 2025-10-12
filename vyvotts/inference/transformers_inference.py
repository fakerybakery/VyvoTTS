from snac import SNAC
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Optional, Dict, Any
import yaml
import time

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


class VyvoTTSTransformersInference:
    """TTS inference engine using Transformers backend."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        model_name: str = "Vyvo/VyvoTTS-LFM2-Neuvillette",
        device: str = "cuda"
    ):
        """Initialize the TTS inference engine.

        Args:
            config: Configuration dictionary containing token constants
            config_path: Path to YAML config file (alternative to config dict)
            model_name: HuggingFace model identifier for the TTS model
            device: Device to run the model on
        """
        # Load configuration
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = load_config(config_path)
        else:
            # Default config path
            default_config_path = "vyvotts/configs/inference/lfm2.yaml"
            self.config = load_config(default_config_path)

        # Set token constants from config
        self.TOKENIZER_LENGTH = self.config['TOKENIZER_LENGTH']
        self.START_OF_TEXT = self.config['START_OF_TEXT']
        self.END_OF_TEXT = self.config['END_OF_TEXT']
        self.START_OF_SPEECH = self.config['START_OF_SPEECH']
        self.END_OF_SPEECH = self.config['END_OF_SPEECH']
        self.START_OF_HUMAN = self.config['START_OF_HUMAN']
        self.END_OF_HUMAN = self.config['END_OF_HUMAN']
        self.START_OF_AI = self.config['START_OF_AI']
        self.END_OF_AI = self.config['END_OF_AI']
        self.PAD_TOKEN = self.config['PAD_TOKEN']
        self.AUDIO_TOKENS_START = self.config['AUDIO_TOKENS_START']

        self.device = device
        self._initialize_models(model_name)

    def _initialize_models(self, model_name: str):
        """Initialize SNAC model, language model, and tokenizer."""
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        self.snac_model = self.snac_model.to(self.device)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="kernels-community/flash-attn3:flash_attention",
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _preprocess_prompts(self, prompts: List[str], chosen_voice: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess prompts by tokenizing, adding special tokens, and padding."""
        if chosen_voice:
            prompts = [f"{chosen_voice}: " + p for p in prompts]

        all_input_ids = []
        for prompt in prompts:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            all_input_ids.append(input_ids)

        start_token = torch.tensor([[self.START_OF_HUMAN]], dtype=torch.int64)
        end_tokens = torch.tensor([[self.END_OF_TEXT, self.END_OF_HUMAN]], dtype=torch.int64)

        all_modified_input_ids = []
        for input_ids in all_input_ids:
            modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
            all_modified_input_ids.append(modified_input_ids)

        all_padded_tensors = []
        all_attention_masks = []
        max_length = max([modified_input_ids.shape[1] for modified_input_ids in all_modified_input_ids])

        for modified_input_ids in all_modified_input_ids:
            padding = max_length - modified_input_ids.shape[1]
            padded_tensor = torch.cat([torch.full((1, padding), self.PAD_TOKEN, dtype=torch.int64), modified_input_ids], dim=1)
            attention_mask = torch.cat([torch.zeros((1, padding), dtype=torch.int64), torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)], dim=1)
            all_padded_tensors.append(padded_tensor)
            all_attention_masks.append(attention_mask)

        all_padded_tensors = torch.cat(all_padded_tensors, dim=0)
        all_attention_masks = torch.cat(all_attention_masks, dim=0)

        input_ids = all_padded_tensors.to(self.device)
        attention_mask = all_attention_masks.to(self.device)

        return input_ids, attention_mask

    def _generate_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                      max_new_tokens: int = 1200, temperature: float = 0.6,
                      top_p: float = 0.95, repetition_penalty: float = 1.1) -> Tuple[torch.Tensor, float]:
        """Generate text using the language model."""
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=1,
                eos_token_id=self.END_OF_SPEECH,
            )

        torch.cuda.synchronize()
        generation_time = time.time() - start_time

        return generated_ids, generation_time

    def _redistribute_codes(self, code_list: List[int]) -> torch.Tensor:
        """Redistribute codes into layers and decode to audio."""
        layer_1 = []
        layer_2 = []
        layer_3 = []
        for i in range((len(code_list)+1)//7):
            layer_1.append(code_list[7*i])
            layer_2.append(code_list[7*i+1]-4096)
            layer_3.append(code_list[7*i+2]-(2*4096))
            layer_3.append(code_list[7*i+3]-(3*4096))
            layer_2.append(code_list[7*i+4]-(4*4096))
            layer_3.append(code_list[7*i+5]-(5*4096))
            layer_3.append(code_list[7*i+6]-(6*4096))

        codes = [torch.tensor(layer_1).unsqueeze(0),
                 torch.tensor(layer_2).unsqueeze(0),
                 torch.tensor(layer_3).unsqueeze(0)]

        codes = [c.to(self.device) for c in codes]
        audio_hat = self.snac_model.decode(codes)
        return audio_hat

    def _parse_output_to_audio(self, generated_ids: torch.Tensor) -> Tuple[List[torch.Tensor], float]:
        """Parse generated token IDs and convert to audio samples."""
        torch.cuda.synchronize()
        start_time = time.time()

        token_to_find = self.START_OF_SPEECH
        token_to_remove = self.END_OF_SPEECH

        token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

        if len(token_indices[1]) > 0:
            last_occurrence_idx = token_indices[1][-1].item()
            cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
        else:
            cropped_tensor = generated_ids

        processed_rows = []
        for row in cropped_tensor:
            masked_row = row[row != token_to_remove]
            processed_rows.append(masked_row)

        code_lists = []
        for row in processed_rows:
            row_length = row.size(0)
            new_length = (row_length // 7) * 7
            trimmed_row = row[:new_length]
            trimmed_row = [t - self.AUDIO_TOKENS_START for t in trimmed_row]
            code_lists.append(trimmed_row)

        my_samples = []
        for code_list in code_lists:
            samples = self._redistribute_codes(code_list)
            my_samples.append(samples)

        torch.cuda.synchronize()
        audio_processing_time = time.time() - start_time

        return my_samples, audio_processing_time

    def generate(self, text: str, voice: Optional[str] = None,
                max_new_tokens: int = 1200, temperature: float = 0.6,
                top_p: float = 0.95, repetition_penalty: float = 1.1) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Generate speech from text input.

        Args:
            text: Input text to convert to speech
            voice: Optional voice identifier
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Penalty for token repetition

        Returns:
            Audio tensor and timing information dictionary
        """
        torch.cuda.synchronize()
        total_start_time = time.time()

        # Preprocessing
        torch.cuda.synchronize()
        preprocess_start = time.time()
        input_ids, attention_mask = self._preprocess_prompts([text], voice)
        torch.cuda.synchronize()
        preprocess_time = time.time() - preprocess_start

        # Text generation
        generated_ids, generation_time = self._generate_text(
            input_ids, attention_mask, max_new_tokens, temperature, top_p, repetition_penalty
        )

        # Audio processing
        audio_samples, audio_processing_time = self._parse_output_to_audio(generated_ids)

        torch.cuda.synchronize()
        total_time = time.time() - total_start_time

        timing_info = {
            'preprocessing_time': preprocess_time,
            'generation_time': generation_time,
            'audio_processing_time': audio_processing_time,
            'total_time': total_time
        }

        return audio_samples[0] if audio_samples else None, timing_info


def main():
    """Example usage of VyvoTTSTransformersInference."""
    engine = VyvoTTSTransformersInference()

    test_text = "Hey there my name is Elise, and I'm a speech generation model that can sound like a person."
    audio, timing_info = engine.generate(test_text)

    if audio is not None:
        print(f"Audio generated successfully with shape: {audio.shape}")
        print(f"Timing info: {timing_info}")
    else:
        print("Failed to generate audio")

    return audio


if __name__ == "__main__":
    main()
