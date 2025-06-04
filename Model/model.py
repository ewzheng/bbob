import torch
import torch.nn as nn
import transformers
import bitsandbytes as bnb
import os
from projector import Projector

class BBOB(nn.Module):
    """
    BBOB multimodal model combining vision encoder, projector, and language model
    """
    
    def __init__(self, base_model, vision_encoder, bnb_config):
        """
        Initialize BBOB model with specified components
        
        Parameters:
            - base_model: path or identifier for base language model
            - vision_encoder: path or identifier for vision encoder
            - bnb_config: quantization configuration for model loading
        """
        super(BBOB, self).__init__()
        # informational print, initialize gpu
        print("Loading BBOB with " + base_model + " and " + vision_encoder + "...\n") 
        n_gpus = torch.cuda.device_count()
        max_memory = torch.cuda.get_device_properties(0).total_memory/1024**2
        print("Max Memory (GB)",max_memory/1024)
        max_memory=f'{max_memory}MB'

        print("Present working Directory",os.getcwd())
        print(f"Number of GPUs: {n_gpus}")

        # initialize components
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(base_model, max_memory=max_memory, quantization_config=bnb_config, device_map="auto")
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)
        self.image_processor = transformers.AutoImageProcessor.from_pretrained(vision_encoder)
        self.vision_encoder = transformers.AutoModel.from_pretrained(vision_encoder)
        self.projector = Projector(self.vision_encoder.config.hidden_size, self.base_model.config.hidden_size)

    ''' Helpers for interacting with internal components'''
    def get_tokenizer(self):
        """
        Get the tokenizer for text processing
        
        Returns:
            - tokenizer: the base model tokenizer
        """
        return self.base_tokenizer
    
    def get_vision_tower(self):
        """
        Get the vision processing components
        
        Returns:
            - tuple: image processor and vision encoder
        """
        return self.image_processor, self.vision_encoder
    
    def freeze_projector(self):
        """
        Freeze projector parameters to prevent training
        """
        self.projector.freeze()
    
    def unfreeze_projector(self):
        """
        Unfreeze projector parameters to enable training
        """
        self.projector.unfreeze()

    def freeze_model(self):
        """
        Freeze base language model parameters
        """
        for weights in self.base_model.parameters():
            weights.requires_grad = False
    
    def unfreeze_model(self):
        """
        Unfreeze base language model parameters
        """
        for weights in self.base_model.parameters():
            weights.requires_grad = True
    
    def freeze_vision_tower(self):
        """
        Freeze vision encoder and image processor parameters
        """
        for weights in self.vision_encoder.parameters():
            weights.requires_grad = False
        for weights in self.image_processor.parameters():
            weights.requires_grad = False

    def unfreeze_vision_tower(self):
        """
        Unfreeze vision encoder and image processor parameters
        """
        for weights in self.vision_encoder.parameters():
            weights.requires_grad = True
        for weights in self.image_processor.parameters():
            weights.requires_grad = True

    def train(self):
        """
        Set all model components to training mode
        """
        self.base_model.train()
        self.vision_encoder.train()
        self.projector.train()

    def eval(self):
        """
        Set all model components to evaluation mode
        """
        self.base_model.eval()
        self.vision_encoder.eval()
        self.projector.eval()
        
    ''' Input processing helpers '''
    def _prepare_visual_inputs(self, vision_in):
        """
        Process raw images into visual tokens
        
        Parameters:
            - vision_in: raw images, image paths, or preprocessed tensors
            
        Returns:
            - visual_tokens: projected visual features ready for LLM
        """
        if vision_in is None:
            return None
            
        # handle different input types - strings, tensors, PIL images
        if isinstance(vision_in, (list, tuple)) and isinstance(vision_in[0], str):
            # image paths or URLs provided
            processed_images = self.image_processor(vision_in, return_tensors="pt")
            pixel_values = processed_images['pixel_values']
        else:
            # assume raw tensors or PIL images
            if not isinstance(vision_in, torch.Tensor):
                processed_images = self.image_processor(vision_in, return_tensors="pt")
                pixel_values = processed_images['pixel_values']
            else:
                pixel_values = vision_in
        
        # move to correct device - match vision encoder
        pixel_values = pixel_values.to(next(self.vision_encoder.parameters()).device)
        
        # extract visual features from vision encoder
        vision_outputs = self.vision_encoder(pixel_values)
        vision_features = vision_outputs.last_hidden_state  # mobilevit returns 4D tensor
        
        # project visual features to text embedding space
        visual_tokens = self.projector(vision_features)  # handles 4D -> 3D conversion internally
        
        return visual_tokens

    def _prepare_text_inputs(self, text_in):
        """
        Process text inputs into embeddings and attention masks
        
        Parameters:
            - text_in: text strings or pre-tokenized input_ids
            
        Returns:
            - text_embeddings: text converted to embedding space
            - attention_mask: mask for padded tokens
        """
        if text_in is None:
            return None, None
            
        # handle string inputs vs pre-tokenized
        if isinstance(text_in, str) or (isinstance(text_in, list) and isinstance(text_in[0], str)):
            # tokenize raw text strings
            text_inputs = self.base_tokenizer(
                text_in, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512  # prevent memory issues
            )
            input_ids = text_inputs['input_ids']
            attention_mask = text_inputs.get('attention_mask', None)
        else:
            # assume already tokenized input_ids
            input_ids = text_in
            attention_mask = None  # will be handled in kwargs
        
        # move to correct device - match base model
        input_ids = input_ids.to(next(self.base_model.parameters()).device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(next(self.base_model.parameters()).device)
        
        # convert token ids to embeddings
        text_embeddings = self.base_model.get_input_embeddings()(input_ids)
        
        return text_embeddings, attention_mask

    def _merge_multimodal_inputs(self, visual_tokens, text_embeddings, attention_mask):
        """
        Combine visual and text inputs for multimodal processing
        
        Parameters:
            - visual_tokens: projected visual features
            - text_embeddings: text embedding vectors  
            - attention_mask: mask for text tokens
            
        Returns:
            - combined_embeddings: concatenated visual + text embeddings
            - combined_attention_mask: extended attention mask for both modalities
        """
        
        # handle different input combinations
        if visual_tokens is not None and text_embeddings is not None:
            # multimodal case - concatenate visual tokens before text
            combined_embeddings = torch.cat([visual_tokens, text_embeddings], dim=1)
            
            # extend attention mask to cover visual tokens
            if attention_mask is not None:
                batch_size = attention_mask.shape[0]
                num_visual_tokens = visual_tokens.shape[1]
                # visual tokens always get attention (no masking)
                visual_attention = torch.ones(
                    batch_size, num_visual_tokens, 
                    device=attention_mask.device, 
                    dtype=attention_mask.dtype
                )
                combined_attention_mask = torch.cat([visual_attention, attention_mask], dim=1)
            else:
                combined_attention_mask = None
                
        elif visual_tokens is not None:
            # vision-only case
            combined_embeddings = visual_tokens
            combined_attention_mask = None  # no masking needed for visual tokens
            
        elif text_embeddings is not None:
            # text-only case  
            combined_embeddings = text_embeddings
            combined_attention_mask = attention_mask
            
        else:
            # error case - need at least one input modality
            raise ValueError("At least one of vision_features or input_ids must be provided")
        
        return combined_embeddings, combined_attention_mask


    ''' Primary API Functions'''

    def generate(self, vision_in=None, text_in=None, max_new_tokens=256, 
                 temperature=0.7, do_sample=True, **kwargs):
        """
        Generate text responses with optional image input
        
        Parameters:
            - vision_in: images or visual input
            - text_in: text prompt/query
            - max_new_tokens: maximum new tokens to generate
            - temperature: sampling temperature
            - do_sample: whether to use sampling
            - return_text: return decoded text vs raw tokens
            
        Returns:
            - generated text or token ids
        """
        
        was_base_training = self.base_model.training
        was_vision_training = self.vision_encoder.training

        self.base_model.eval()
        self.vision_encoder.eval()

        # process visual inputs through vision tower and projector
        visual_tokens = self._prepare_visual_inputs(vision_in)
        
        # process text inputs through tokenizer and embedding layer
        text_embeddings, attention_mask = self._prepare_text_inputs(text_in)
        
        # combine visual and text modalities
        combined_embeddings, combined_attention_mask = self._merge_multimodal_inputs(
            visual_tokens, text_embeddings, attention_mask
        )
        
        # setup generation parameters with defaults
        generation_kwargs = {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'do_sample': do_sample,
            'pad_token_id': self.base_tokenizer.pad_token_id,
            'eos_token_id': self.base_tokenizer.eos_token_id,
            **kwargs
        }
        
        # generate tokens without computing gradients
        with torch.no_grad():
            outputs = self.base_model.generate(
                inputs_embeds=combined_embeddings,
                attention_mask=combined_attention_mask,
                **generation_kwargs
            )
        
        # convert token ids back to readable text
        generated_text = self.base_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # restore model state
        if was_base_training:
            self.base_model.train()
        if was_vision_training:
            self.vision_encoder.train()

        return generated_text

    def forward(self, vision_features=None, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        Main forward pass for multimodal model using preprocessed inputs
        
        Parameters:
            - vision_features: precomputed vision encoder features [batch, seq_len, hidden_dim]
            - input_ids: preprocessed text token ids [batch, seq_len]
            - attention_mask: preprocessed attention mask [batch, seq_len]
            - labels: ground truth labels for loss computation
            
        Returns:
            - model outputs including logits and loss
        """
        
        # project vision features to text embedding space
        if vision_features is not None:
            vision_features = vision_features.to(next(self.projector.parameters()).device)
            visual_tokens = self.projector(vision_features)
        else:
            visual_tokens = None
        
        # convert token ids to text embeddings
        if input_ids is not None:
            input_ids = input_ids.to(next(self.base_model.parameters()).device)
            text_embeddings = self.base_model.get_input_embeddings()(input_ids)
        else:
            text_embeddings = None
            
        # move attention mask to correct device
        if attention_mask is not None:
            attention_mask = attention_mask.to(next(self.base_model.parameters()).device)
        
        # combine visual and text modalities
        combined_embeddings, combined_attention_mask = self._merge_multimodal_inputs(
            visual_tokens, text_embeddings, attention_mask
        )
        
        # forward through base language model
        outputs = self.base_model(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            labels=labels,
            **kwargs
        )
        
        return outputs
    
    def save_pretrained(self, save_directory, save_base_model=False, save_vision_encoder=False):
        """
        Save model components to directory
        
        Parameters:
            - save_directory: path to save model files
            - save_base_model: whether to save the base LLM (usually unnecessary)
            - save_vision_encoder: whether to save vision encoder (usually unnecessary)
        """
        import json
        import os
        
        # create save directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)
        
        # save model configuration
        config = {
            'model_type': 'BBOB',
            'base_model': self.base_model.config.name_or_path if hasattr(self.base_model.config, 'name_or_path') else 'unknown',
            'vision_encoder': self.vision_encoder.config.name_or_path if hasattr(self.vision_encoder.config, 'name_or_path') else 'unknown', 
            'projector_config': {
                'input_dim': self.projector.net[0].in_features,
                'output_dim': self.projector.net[0].out_features,
                'hidden_dim': self.projector.net[2].out_features
            },
            'vision_hidden_size': self.vision_encoder.config.hidden_size,
            'text_hidden_size': self.base_model.config.hidden_size
        }
        
        with open(os.path.join(save_directory, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        # always save the projector - this is our custom component
        projector_path = os.path.join(save_directory, 'projector.pth')
        torch.save(self.projector.state_dict(), projector_path)
        print(f"Saved projector to {projector_path}")
        
        # optionally save base model (usually unnecessary as it's a standard model)
        if save_base_model:
            base_model_path = os.path.join(save_directory, 'base_model')
            self.base_model.save_pretrained(base_model_path)
            self.base_tokenizer.save_pretrained(base_model_path)
            print(f"Saved base model to {base_model_path}")
        
        # optionally save vision encoder (usually unnecessary as it's a standard model)
        if save_vision_encoder:
            vision_path = os.path.join(save_directory, 'vision_encoder')
            self.vision_encoder.save_pretrained(vision_path)
            self.image_processor.save_pretrained(vision_path)
            print(f"Saved vision encoder to {vision_path}")
            
        print(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, model_path, base_model=None, vision_encoder=None, 
                       load_base_model=False, load_vision_encoder=False):
        """
        Load model from saved directory
        
        Parameters:
            - model_path: path to saved model directory
            - base_model: base model name/path (if not loading from saved)
            - vision_encoder: vision encoder name/path (if not loading from saved)
            - load_base_model: whether to load base model from saved files
            - load_vision_encoder: whether to load vision encoder from saved files
            
        Returns:
            - loaded BBOB model instance
        """
        import json
        import os
        
        # load configuration
        config_path = os.path.join(model_path, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # determine which models to use
        if load_base_model:
            # load base model from saved files
            base_model_to_use = os.path.join(model_path, 'base_model')
        else:
            # use provided base model or fallback to config
            base_model_to_use = base_model or config.get('base_model', 'microsoft/DialoGPT-medium')
            
        if load_vision_encoder:
            # load vision encoder from saved files  
            vision_encoder_to_use = os.path.join(model_path, 'vision_encoder')
        else:
            # use provided vision encoder or fallback to config
            vision_encoder_to_use = vision_encoder or config.get('vision_encoder', 'apple/mobilevit-small')
        
        # create model instance with specified components
        print(f"Loading BBOB from {model_path}")
        print(f"Base model: {base_model_to_use}")
        print(f"Vision encoder: {vision_encoder_to_use}")
        
        model = cls(
            base_model=base_model_to_use,
            vision_encoder=vision_encoder_to_use,
            bnb_config=None  # TODO: add bnb_config parameter if needed
        )
        
        # load projector weights
        projector_path = os.path.join(model_path, 'projector.pth')
        if os.path.exists(projector_path):
            projector_state = torch.load(projector_path, map_location='cpu')
            model.projector.load_state_dict(projector_state)
            print(f"Loaded projector from {projector_path}")
        else:
            print(f"Warning: No projector weights found at {projector_path}")
        
        return model

    def save_projector_only(self, save_path):
        """
        Save only the projector weights for quick fine-tuning
        
        Parameters:
            - save_path: path to save projector weights (.pth file)
        """
        torch.save(self.projector.state_dict(), save_path)
        print(f"Projector saved to {save_path}")
        
    def load_projector_weights(self, weights_path):
        """
        Load projector weights from file
        
        Parameters:
            - weights_path: path to projector weights file
        """
        projector_state = torch.load(weights_path, map_location='cpu')
        self.projector.load_state_dict(projector_state)
        print(f"Loaded projector weights from {weights_path}")
        