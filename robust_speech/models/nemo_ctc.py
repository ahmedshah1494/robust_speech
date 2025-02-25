from robust_speech.models.base_robust_speech_model import BaseASR
import torch

class NeMoCTCASR(BaseASR):
    """
    HuggingFace ASR model
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def eval_forward(self,
                     feats=None,
                     feat_lens=None,
                     tokens=None,
                     tok_lens=None,
                     options=None,
                     loss_options=None):
        model = self.modules.model
        dtype = torch.float16 if options.get("fp16", False) else torch.float32
        logits, logits_len, greedy_predictions = model.forward(
            processed_signal=feats, processed_signal_length=feat_lens,
        )
        loss_value = model.loss(
            log_probs=logits, targets=tokens, input_lengths=logits_len, target_lengths=tok_lens
        )

        # Add auxiliary losses, if registered
        loss_value = model.add_auxiliary_losses(loss_value)
        
        current_hypotheses, all_hyp = model.decoding.ctc_decoder_predictions_tensor(
            logits, decoder_lengths=logits_len, return_hypotheses=False,
        )

        if all_hyp is None:
            hypotheses = current_hypotheses
        else:
            hypotheses = all_hyp

        return loss_value, hypotheses
    
    def train_attack_forward(self,
                     feats=None,
                     feat_lens=None,
                     tokens=None,
                     tok_lens=None,
                     options=None,
                     loss_options=None):
        model = self.modules.model.eval()
        dtype = torch.float16 if options.get("fp16", False) else torch.float32
        logits, logits_len, greedy_predictions = model.forward(
            processed_signal=feats, processed_signal_length=feat_lens,
        )
        loss_value = model.loss(
            log_probs=logits, targets=tokens, input_lengths=logits_len, target_lengths=tok_lens
        )

        # Add auxiliary losses, if registered
        loss_value = model.add_auxiliary_losses(loss_value)

        return loss_value, greedy_predictions

    def text_to_tokens(self, batch):
        tokenizer = self.modules.model.tokenizer
        wrd = batch.wrd
        tokens = [tokenizer.text_to_ids(w) for w in wrd]
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