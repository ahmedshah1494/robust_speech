
import logging
import os
import sys

import speechbrain as sb
import torch
import torch.nn as nn
import string
from transformers import PreTrainedModel, WhisperProcessor, Wav2Vec2ForCTC
import librosa

import robust_speech as rs
from robust_speech.adversarial.brain import AdvASRBrain
from typing import Optional, Union

logger = logging.getLogger(__name__)


# Define training procedure
class CanaryASR(AdvASRBrain):
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
                     wavs=None,
                     wav_lens=None,
                     feats=None,
                     feat_lens=None,
                     tokens=None,
                     tok_lens=None,
                     options=None,
                     loss_options=None):
        model = self.modules.model
        dtype = torch.float16 if options.get("fp16", False) else torch.float32
        if wavs is not None:
            wavs = wavs.to(dtype)
        if feats is not None:
            feats = feats.to(dtype)
        # with torch.no_grad():
        #     result = model(
        #         wavs.to(dtype), labels=tokens, **loss_options, **options)
        #     loss = result["loss"].detach()
        #     #logits = result["logits"]
        #     #pred_tokens = logits.argmax(dim=-1)
        #     if model.can_generate():
        #         genkwargs = {}
        #         if isinstance(self.hparams.processor, WhisperProcessor):
        #             lang = getattr(self.hparams, "language", "english")
        #             genkwargs['forced_decoder_ids'] = self.hparams.processor.get_decoder_prompt_ids(language=lang, task="transcribe")
        #         pred_tokens = model.generate(wavs.to(dtype), **options, **genkwargs)
        #     else:
        #         pred_tokens = result["logits"].argmax(dim=-1)
        # return loss.reshape(-1), pred_tokens
        input_ids, labels = tokens[:, :-1], tokens[:, 1:]
        # print(input_ids)
        log_probs, encoded_len, enc_states, enc_mask = model.forward(
            input_signal=wavs, input_signal_length=wav_lens,
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
                     wavs=None,
                     wav_lens=None,
                     feats=None,
                     feat_lens=None,
                     tokens=None,
                     tok_lens=None,
                     options=None,
                     loss_options=None):
        # model: PreTrainedModel = self.modules.model
        # result = model(wavs, labels=tokens, **loss_options, **options)
        # loss = result["loss"]
        # #logits = self.modules.whisper.model.transcribe(wavs[0], beam_size=1)
        # logits = result["logits"]
        # pred_tokens = logits.argmax(dim=-1)
        # return loss.reshape(-1), pred_tokens

        model = self.modules.model.eval()
        dtype = torch.float16 if options.get("fp16", False) else torch.float32
        input_ids, labels = tokens[:, :-1], tokens[:, 1:]
        if wavs is not None:
            wavs = wavs.to(dtype)
        if feats is not None:
            feats = feats.to(dtype)
        log_probs, encoded_len, enc_states, enc_mask = model.forward(
            input_signal=wavs, input_signal_length=wav_lens,
            processed_signal=feats, processed_signal_length=feat_lens,
            transcript=input_ids, transcript_length=tok_lens
        )
        loss = model.loss(log_probs=log_probs, labels=labels)

        pred_tokens = log_probs.argmax(dim=-1)
        return loss, pred_tokens

    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        if not stage == sb.Stage.TRAIN:
            self.modules = self.modules.eval()
        if not stage == rs.Stage.ATTACK:
            batch = batch.to(self.device)
        wavs, _ = batch.sig
        wav_lens = torch.LongTensor([len(w) for w in wavs]).to(self.device)
        tokenizer = self.modules.model.tokenizer
        wrd = batch.wrd
        tokens = [tokenizer.text_to_ids(w, getattr(self.hparams, 'language', 'en')) for w in wrd]
        tok_len = torch.LongTensor([len(t) for t in tokens]).to(self.device)
        max_len = max(tok_len)
        tokens = torch.stack([torch.nn.functional.pad(torch.LongTensor(t), (0, max_len - len(t)), value=tokenizer.pad) for t in tokens]).to(self.device)
        tokens_bos = tokenizer.bos


        # Add augmentation if specified
        options = {}
        loss_options = {}
        if options.get("fp16", False):
            self.modules.to(torch.float16)
        dtype = torch.float16 if options.get("fp16", False) else torch.float32

        if hasattr(self.hparams, "smoothing") and self.hparams.smoothing:
            wavs = self.hparams.smoothing(wavs, wav_lens)
        if stage == sb.Stage.TRAIN or stage == rs.Stage.ATTACK:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)

            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)
        
        feats, feat_lens = self.hparams.compute_features(wavs, wav_lens) if self.hparams.compute_features is not None else wavs
        # Forward pass
        if stage != sb.Stage.TRAIN and stage != rs.Stage.ATTACK:
            # Decode token terms to words
            # print(wrd, tokens, tok_len)
            loss, pred_tokens = self.eval_forward(
                feats=feats,
                feat_lens=feat_lens,
                tokens=tokens,
                tok_lens=tok_len,
                options=options,
                loss_options=loss_options
            )
        else:
            loss, pred_tokens = self.train_attack_forward(
                feats=feats,
                feat_lens=feat_lens,
                tokens=tokens,
                tok_lens=tok_len,
                options=options,
                loss_options=loss_options
            )
        return loss, pred_tokens, stage

    def get_tokens(self, predictions):
        #text = predictions[1]["text"]
        #tokens = torch.LongTensor([self.tokenizer.encode(text)])
        if predictions[2] in [sb.Stage.VALID, sb.Stage.TEST]:
            tokens = predictions[1].cpu()
        else:
            tokens = predictions[1][:, :-1].cpu()
        return tokens

    def compute_objectives(
        self, predictions, batch, stage, adv=False, targeted=False, reduction="mean"
    ):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        loss, predicted, save_stage = predictions

        ids = batch.id
        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens, tokens_lens = batch.tokens
            tokens_eos, tokens_eos_lens = batch.tokens_eos
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens, tokens_eos_lens], dim=0)
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)

        if stage != sb.Stage.TRAIN and stage != rs.Stage.ATTACK:
            # Decode token terms to words
            # predicted = [self.tokenizer.decode(t, skip_special_tokens=True).strip().upper().translate(
            #     str.maketrans('', '', string.punctuation)) for t in pred_tokens]
            # print(pred_tokens.shape)
            # predicted_words = self.tokenizer.batch_decode(pred_tokens, skip_special_tokens=True).strip().upper().translate(
            #     str.maketrans('', '', string.punctuation))
            # predicted_words = [self.tokenizer.decode(
            #    t).strip() for t in pred_tokens]
            predicted = [wrd.upper().translate(str.maketrans(
                '', '', string.punctuation)) for wrd in predicted]
            predicted_words = [wrd.split(" ") for wrd in predicted]
            target = [wrd.upper().translate(str.maketrans(
                '', '', string.punctuation)) for wrd in batch.wrd]
            target_words = [wrd.split(" ") for wrd in target]
            #target_words = [wrd.split(" ") for wrd in batch.wrd]
            # print(predicted_words, target_words)

            if adv:
                if targeted:
                    self.adv_wer_metric_target.append(
                        ids, predicted_words, target_words
                    )
                    self.adv_cer_metric_target.append(
                        ids, predicted, target
                    )
                    self.adv_ser_metric_target.append(
                        ids, predicted, target)
                else:
                    self.adv_wer_metric.append(
                        ids, predicted_words, target_words)
                    self.adv_cer_metric.append(
                        ids, predicted_words, target_words)
                print('adv_cer =', self.adv_cer_metric.summarize())
            else:
                self.wer_metric.append(ids, predicted_words, target_words)
                self.cer_metric.append(ids, predicted_words, target_words)
                print('cer =', self.cer_metric.summarize())
            # if adv and targeted:
                # print(" ".join(predicted_words[0]))
                #print(" ".join(target_words[0]))
        if reduction == 'mean':
            loss = loss.mean()
        return loss

    def init_optimizers(self):
        "Initializes the optimizer and model optimizer"
        self.optimizer = self.hparams.opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("optimizer", self.optimizer)
    
    def on_stage_end(self, stage, stage_loss, epoch, stage_adv_loss=None, stage_adv_loss_target=None):
        super().on_stage_end(stage, stage_loss, epoch, stage_adv_loss, stage_adv_loss_target)
        if stage == sb.Stage.TEST:
            with open(self.hparams.wer_file.replace("wer", "cer"), "w") as cer:
                self.cer_metric.write_stats(cer)
            with open(f'{self.hparams.wer_file.replace("wer", "cer_adv")}', "w") as cer:
                self.adv_cer_metric.write_stats(cer)
            with open(f'{self.hparams.wer_file.replace("wer", "wer_adv")}', "w") as wer:
                self.adv_wer_metric.write_stats(wer)
