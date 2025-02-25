from robust_speech.models.base_robust_speech_model import BaseASR
import torch

class NeMoRNNTASR(BaseASR):
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
        encoded, encoded_len = model.forward(
            processed_signal=feats, processed_signal_length=feat_lens,
        )
        best_hyp, _ = model.decoding.rnnt_decoder_predictions_tensor(
                        encoded,
                        encoded_len,
                        return_hypotheses=False,
                        partial_hypotheses=None,
                    )
        return torch.zeros(feats.shape[0], device=feats.device), best_hyp
    
    def train_attack_forward(self,
                     feats=None,
                     feat_lens=None,
                     tokens=None,
                     tok_lens=None,
                     options=None,
                     loss_options=None):
        model = self.modules.model
        dtype = torch.float16 if options.get("fp16", False) else torch.float32
        feats = feats.to(dtype)
        encoded, encoded_len = model.forward(
            processed_signal=feats, processed_signal_length=feat_lens,
        )
        decoder, target_length, states = model.decoder(targets=tokens, target_length=tok_lens)
        loss, _, _, _ = model.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=tokens,
                transcript_lengths=tok_lens,
                compute_wer=False,
            )
        return loss, encoded.argmax(dim=1)
    
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