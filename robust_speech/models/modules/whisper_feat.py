import torch
import librosa
from typing import Optional, Union

class WhisperLogMelSpectrogram(object):
    def __init__(self,
                n_mels: int = 80,
                n_fft: int = 400,
                hop_length: int = 160,
                padding: int = 480000,
                chunk_size: int = 3000,
                device: Optional[Union[str, torch.device]] = None,) -> None:
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.padding = padding
        self.chunk_size = chunk_size
        self.device = device
    
    def __call__(self,audio: torch.Tensor):
        """
        Compute the log-Mel spectrogram of

        Parameters
        ----------
        audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
            The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

        n_mels: int
            The number of Mel-frequency filters, only 80 is supported

        padding: int
            Number of zero samples to pad to the right

        device: Optional[Union[str, torch.device]]
            If given, the audio tensor is moved to this device before STFT

        Returns
        -------
        torch.Tensor, shape = (80, n_frames)
            A Tensor that contains the Mel spectrogram
        """
        def mel_filters(device, n_mels: int = 80) -> torch.Tensor:
            """
            load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
            Allows decoupling librosa dependency; saved using:

                np.savez_compressed(
                    "mel_filters.npz",
                    mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
                )
            """
            assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
            mels = librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)
            return torch.from_numpy(mels).to(device)
        
        if self.device is not None:
            audio = audio.to(self.device)
        if self.padding > 0:
            audio = torch.nn.functional.pad(audio, (0, self.padding))
        window = torch.hann_window(self.n_fft).to(audio.device)
        stft = torch.stft(audio, self.n_fft, self.hop_length, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2

        filters = mel_filters(audio.device, self.n_mels)
        mel_spec = filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        log_spec = log_spec[..., :self.chunk_size]
        return log_spec