from robust_speech.models.base_robust_speech_model import BaseASR
from transformers import WhisperProcessor, Wav2Vec2ForCTC
import torch

class HuggingFaceASR(BaseASR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.modules.model, Wav2Vec2ForCTC) and hasattr(self.hparams, "language"):
            self.modules.model.load_adapter(self.hparams.language, force_reload=False)
    
    def eval_forward(self,
                     feats=None,
                     feat_lens=None,
                     tokens=None,
                     tok_lens=None,
                     options=None,
                     loss_options=None):
        '''
        Forward pass for evaluation
        '''
        model = self.modules.model
        dtype = torch.float16 if options.get("fp16", False) else torch.float32
        with torch.no_grad():
            result = model(
                feats.to(dtype), labels=tokens, **loss_options, **options)
            loss = result["loss"].detach()
            if model.can_generate():
                genkwargs = {}
                if isinstance(self.hparams.processor, WhisperProcessor):
                    lang = getattr(self.hparams, "language", "english")
                    genkwargs['forced_decoder_ids'] = self.hparams.processor.get_decoder_prompt_ids(language=lang, task="transcribe")
                pred_tokens = model.generate(feats.to(dtype), **options, **genkwargs)
            else:
                pred_tokens = result["logits"].argmax(dim=-1)
        
        tokenizer = self.hparams.tokenizer
        predicted = [tokenizer.decode(t, skip_special_tokens=True) for t in pred_tokens]
        return loss.reshape(-1), predicted
    
    def train_attack_forward(self,
                     feats=None,
                     feat_lens=None,
                     tokens=None,
                     tok_lens=None,
                     options=None,
                     loss_options=None):
        '''
        Forward pass for training and attack
        '''
        model = self.modules.model
        result = model(feats, labels=tokens, **loss_options, **options)
        loss = result["loss"]
        logits = result["logits"]
        pred_tokens = logits.argmax(dim=-1)
        tokenizer = self.hparams.tokenizer
        predicted = [tokenizer.decode(t, skip_special_tokens=True) for t in pred_tokens]
        return loss.reshape(-1), predicted
    
    def text_to_tokens(self, batch):
        if hasattr(batch, "tokens"):
            tokens, _ = batch.tokens
        else:
            # Else it will contain the text that can be tokenized using the 
            # tokenizer defined in the model config.
            wrd = batch.wrd
            tokenizer = self.hparams.tokenizer
            tokens_list = [tokenizer.encode(w) for w in wrd]
            tokens = torch.LongTensor(tokens_list).to(self.device)
        return tokens, None
    
    def wav_to_feats(self, batch):
        wavs, _ = batch.sig
        feats = self.hparams.compute_features(wavs) if self.hparams.compute_features is not None else wavs
        return feats, None