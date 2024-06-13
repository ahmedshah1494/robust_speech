from robust_speech.models.base_robust_speech_model import BaseASR
import torch

class CanaryASR(BaseASR):
    """
    HuggingFace ASR model
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (not hasattr(self.hparams, "language")) or (self.hparams.language == 'en'):
            lang_id = 4
        elif self.hparams.language == 'es':
            lang_id = 5
        else:
            raise ValueError(f'Unsupported language {self.hparams.language}')
        
        self.decoder_input_ids = torch.LongTensor([[3,   lang_id,   8,   lang_id, 11, 137, 327,   2]])


    def eval_forward(self,
                     feats=None,
                     feat_lens=None,
                     tokens=None,
                     tok_lens=None,
                     options=None,
                     loss_options=None):
        model = self.modules.model
        dtype = torch.float16 if options.get("fp16", False) else torch.float32
        feats = feats.to(dtype)
        input_ids, labels = tokens[:, :-1], tokens[:, 1:]
        log_probs, encoded_len, enc_states, enc_mask = model.forward(
            input_signal=None, input_signal_length=None,
            processed_signal=feats, processed_signal_length=feat_lens,
            transcript=input_ids, transcript_length=tok_lens
        )
        loss = model.loss(log_probs=log_probs, labels=labels)
        beam_hypotheses = model.decoding.decode_predictions_tensor(
            encoder_hidden_states=enc_states,
            encoder_input_mask=enc_mask,
            decoder_input_ids=self.decoder_input_ids[:, : model.context_len_for_AR_decoding].to(self.device)
            if model.context_len_for_AR_decoding > 0
            else None,
            return_hypotheses=False,
        )[0]
        beam_hypotheses = [model.decoding.strip_special_tokens(text) for text in beam_hypotheses]

        return loss, beam_hypotheses
    
    def train_attack_forward(self,
                     feats=None,
                     feat_lens=None,
                     tokens=None,
                     tok_lens=None,
                     options=None,
                     loss_options=None):
        model = self.modules.model.eval()
        dtype = torch.float16 if options.get("fp16", False) else torch.float32
        input_ids, labels = tokens[:, :-1], tokens[:, 1:]
        feats = feats.to(dtype)
        log_probs, encoded_len, enc_states, enc_mask = model.forward(
            input_signal=None, input_signal_length=None,
            processed_signal=feats, processed_signal_length=feat_lens,
            transcript=input_ids, transcript_length=tok_lens
        )
        loss = model.loss(log_probs=log_probs, labels=labels)

        pred_tokens = log_probs.argmax(dim=-1)
        return loss, pred_tokens

    def text_to_tokens(self, batch):
        tokenizer = self.modules.model.tokenizer
        wrd = batch.wrd
        tokens = [tokenizer.text_to_ids(w, getattr(self.hparams, 'language', 'en')) for w in wrd]
        tok_len = torch.LongTensor([len(t) for t in tokens]).to(self.device)
        max_len = max(tok_len)
        tokens = torch.stack([torch.nn.functional.pad(torch.LongTensor(t), (0, max_len - len(t)), value=tokenizer.pad) for t in tokens]).to(self.device)
        return tokens, tok_len

    def wav_to_feats(self, batch):
        wavs, _ = batch.sig
        wav_lens = torch.LongTensor([len(w) for w in wavs]).to(self.device)
        wavs, _ = batch.sig
        feats, feat_lens = self.hparams.compute_features(wavs, wav_lens)
        return feats, feat_lens