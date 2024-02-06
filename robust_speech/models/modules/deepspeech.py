"""A CTC ASR system with librispeech supporting adversarial attacks.
The system can employ any encoder. Decoding is performed with
ctc greedy decoder.
The neural network is trained on CTC likelihood target and character units
are used as basic recognition tokens. Training is performed on the full
LibriSpeech dataset (960 h).

Inspired from both SpeechBrain Wav2Vec2
(https://github.com/speechbrain/speechbrain/blob/develop/recipes/LibriSpeech/ASR/CTC/train_with_wav2vec.py)
and Seq2Seq 
(https://github.com/speechbrain/speechbrain/blob/develop/recipes/LibriSpeech/ASR/seq2seq/train.py)

"""

import logging
import string

import speechbrain as sb
import torch

import robust_speech as rs
from robust_speech.models.ctc import CTCASR
from torchaudio.transforms import Spectrogram
from torchaudio.functional import spectrogram
from deepspeech_pytorch.model import DeepSpeech
from deepspeech_pytorch.decoder import GreedyDecoder

logger = logging.getLogger(__name__)


# Define training procedure
class DeepspeechCTCASR(CTCASR):
    """
    Encoder-only CTC model.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    #     self.tokenizer = sb.dataio.encoder.CTCTextEncoder()
    #     self.tokenizer.update_from_iterable(self.modules.model.labels)
    #     self.tokenizer.insert_blank('_', 0)
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        if not stage == rs.Stage.ATTACK:
            batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        # wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        # Add augmentation if specified

        if hasattr(self.hparams, "smoothing") and self.hparams.smoothing:
            wavs = self.hparams.smoothing(wavs, wav_lens)

        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)

            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)
        # Forward pass
        feats, lens = self.hparams.compute_features(
            wavs) if self.hparams.compute_features is not None else wavs
        # if stage == sb.Stage.TRAIN:
        #     feats = self.modules.normalize(feats, wav_lens)
        # else:
        #     # don't update normalization outside of training!
        #     feats = self.modules.normalize(
        #         feats, wav_lens, epoch=self.modules.normalize.update_until_epoch + 1
        #     )
        if stage == rs.Stage.ATTACK:
            p_ctc, out_lens, _ = self.modules.model(feats, lens)
        else:
            p_ctc, out_lens, _ = self.modules.model(feats.detach(), lens.detach())
        out_lens_frac = out_lens.float() / p_ctc.shape[1]

        if stage not in [sb.Stage.TRAIN, rs.Stage.ATTACK]:
            p_tokens, _ = self.hparams.decoder.decode(
                p_ctc, out_lens
            )
        else:
            p_tokens = None
        return p_ctc, p_tokens, out_lens_frac

    def compute_objectives(
        self, predictions, batch, stage, adv=False, targeted=False, reduction="mean"
    ):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        p_ctc, predicted, wav_lens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens, tokens_eos_lens], dim=0)
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)

        
        loss_ctc = self.hparams.ctc_cost(
            p_ctc, tokens, wav_lens, tokens_lens, reduction=reduction
        )
        loss = loss_ctc

        if stage != sb.Stage.TRAIN and stage != rs.Stage.ATTACK:
            # Decode token terms to words
            predicted = [p[0] for p in predicted]
            predicted_words = [p.split() for p in predicted]
            target = [wrd.upper().translate(str.maketrans(
                '', '', string.punctuation)) for wrd in batch.wrd]
            target_words = [wrd.split(" ") for wrd in batch.wrd]

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

        return loss
    
    def on_stage_end(self, stage, stage_loss, epoch, stage_adv_loss=None, stage_adv_loss_target=None):
        super().on_stage_end(stage, stage_loss, epoch, stage_adv_loss, stage_adv_loss_target)
        if stage == sb.Stage.TEST:
            with open(self.hparams.wer_file.replace("wer", "cer"), "w") as cer:
                self.cer_metric.write_stats(cer)
            with open(f'{self.hparams.wer_file.replace("wer", "cer_adv")}', "w") as cer:
                self.adv_cer_metric.write_stats(cer)
            with open(f'{self.hparams.wer_file.replace("wer", "wer_adv")}', "w") as wer:
                self.adv_wer_metric.write_stats(wer)

def get_deepspeech_model():    
    model = DeepSpeech.load_from_checkpoint('/ocean/projects/cis220031p/mshah1/audio_robustness_benchmark/deepspeech_ckps/librispeech_pretrained_v3.ckpt')
    return model

class DeepSpeechFeatureExtractor(torch.nn.Module):
    def __init__(self, spec_cfg, normalize):
        super().__init__()
        self.spec_cfg = spec_cfg
        self.normalize = normalize
        # self.spec = Spectrogram(
        #     n_fft=int(spec_cfg['sample_rate']*spec_cfg['window_size']),
        #     hop_length=int(spec_cfg['sample_rate']*spec_cfg['window_stride']),
        #     win_length=int(spec_cfg['sample_rate']*spec_cfg['window_size']),
        #     window_fn=torch.hamming_window,
        # )
        # self.normalize = normalize

    def forward(self, audio):
        if isinstance(audio, list):
            # pad batch of waveforms to the same length
            audio = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True)
        spec = spectrogram(
            audio,
            0,
            torch.hamming_window(int(self.spec_cfg['sample_rate']*self.spec_cfg['window_size']), device=audio.device),
            int(self.spec_cfg['sample_rate']*self.spec_cfg['window_size']),
            int(self.spec_cfg['sample_rate']*self.spec_cfg['window_stride']),
            int(self.spec_cfg['sample_rate']*self.spec_cfg['window_size']),
            2.,
            False,
        )
        spec = torch.log1p(spec)
        if self.normalize:
            spec = (spec - spec.mean()) / spec.std()
        spec = spec.unsqueeze(1)
        return spec, torch.LongTensor([spec.shape[-1]])

def get_decoder(model):
    decoder = GreedyDecoder(model.labels)
    return decoder