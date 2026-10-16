import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import fluidsynth
import time
import argparse
from pathlib import Path
from sf2utils.sf2parse import Sf2File

# ==========================================
# 1. Psychoacoustic roughness kernel R(f1, f2) - Plomp-Levelt model
# ==========================================
def plomp_levelt_raw(f1, f2):
    """
    Calculate the roughness (dissonance) between two pure tones (f1, f2).
    """
    f_min = np.minimum(f1, f2)
    f_diff = np.abs(f1 - f2)
    
    # Approximation of the critical bandwidth.
    cbw = 24.7 * (4.37 * f_min / 1000.0 + 1.0)
    
    # Relative frequency difference x.
    x = f_diff / cbw
    
    # Plomp-Levelt fit: a * exp(-b*x) - a * exp(-c*x).
    # The roughness peak occurs near x ≈ 0.25.
    a, b, c = 3.5, 3.5, 5.75
    r = a * (np.exp(-b * x) - np.exp(-c * x))
    r[x < 0] = 0
    return np.maximum(r, 0)


# ==========================================
# 2. Render notes from an SF2 file and extract spectra S(f_base, f)
# ==========================================
class SF2SpectrumExtractor:
    def __init__(self, sf2_path, sample_rate=44100):
        self.sf2_path = sf2_path
        self.sr = sample_rate
        self.fs = fluidsynth.Synth(samplerate=int(sample_rate))
        self.sfid = self.fs.sfload(self.sf2_path)
        self.sf2_handle = open(self.sf2_path, "rb")
        self.sf2_parser = Sf2File(self.sf2_handle)
        bank, preset, preset_name, midi_range = self._find_preset()
        if self.fs.program_select(0, self.sfid, bank, preset) != 0:
            raise RuntimeError(
                f"Unable to select SF2 preset: bank={bank}, preset={preset}, "
                f"name={preset_name!r}"
            )
        self.midi_range = midi_range
        print(f"Selected SF2 preset: bank={bank}, preset={preset}, name={preset_name}")
        print(f"Declared SF2 MIDI key range: {midi_range.start}–{midi_range.stop}")

    def _find_preset(self):
        """Select the default preset and read its SF2 keyRange."""
        default_name = self.fs.sfpreset_name(self.sfid, 0, 0)
        candidates = [(0, 0, default_name)] if default_name is not None else []
        for bank in range(128):
            for preset in range(128):
                if (bank, preset) == (0, 0):
                    continue
                preset_name = self.fs.sfpreset_name(self.sfid, bank, preset)
                if preset_name is not None:
                    candidates.append((bank, preset, preset_name))
        for bank, preset, preset_name in candidates:
            for sf2_preset in self.sf2_parser.presets:
                if (
                    getattr(sf2_preset, "bank", None) == bank
                    and getattr(sf2_preset, "preset", None) == preset
                ):
                    return bank, preset, preset_name, sf2_preset.key_range

        raise RuntimeError(f"No usable preset found in SF2 file: {self.sf2_path}")

    def get_spectrum(self, midi_note, num_samples=22050):
        """
        Trigger a MIDI note and extract its steady-state spectrum.
        """
        self.fs.noteon(0, midi_note, 100)
        # Warm up briefly to avoid the initial transient.
        _ = self.fs.get_samples(4410) 
        # Extract the steady-state waveform.
        samples = self.fs.get_samples(num_samples)
        self.fs.noteoff(0, midi_note)
        
        # Mix stereo channels down to mono.
        samples_mono = samples[0::2].astype(np.float64) / 32768.0
        
        # Apply a window and compute the FFT.
        window = np.hanning(len(samples_mono))
        fft_data = np.abs(np.fft.rfft(samples_mono * window))
        freqs = np.fft.rfftfreq(len(samples_mono), 1.0 / self.sr)
        
        # Normalize the magnitude spectrum.
        max_val = np.max(fft_data)
        if max_val > 0:
            fft_data /= max_val
            
        return freqs, fft_data


# ==========================================
# 3. Calculate the two-note dissonance D(fA, fB)
# ==========================================
def calculate_dissonance_matrix(extractor, midi_range=range(36, 85), freq_max=10000):
    """
    Iterate over the MIDI note range and calculate the 2D dissonance matrix
    D[A, B].
    """
    notes = list(midi_range)
    N = len(notes)
    
    # Extract all note spectra once to avoid repeated computation.
    print("Extracting note spectra from the SF2 file...")
    spectra = []
    for note in notes:
        freqs, spec = extractor.get_spectrum(note)
        # Limit high frequencies to reduce unnecessary computation.
        valid_idx = freqs <= freq_max
        # Remove low-energy frequency bins as a noise filter.
        f_active = freqs[valid_idx]
        s_active = spec[valid_idx]
        mask = s_active > 0.01  # Ignore components below -40 dB.
        
        spectra.append((f_active[mask], s_active[mask]))

    D = np.zeros((N, N))
    print("Calculating the two-note dissonance matrix D(f_A, f_B)...")
    start_time = time.time()
    
    # Compute pairwise discrete interference (a sampled continuous integral).
    for i in range(N):
        fA_vec, sA_vec = spectra[i]
        for j in range(i, N):
            fB_vec, sB_vec = spectra[j]
            
            # Build a grid of all frequency-bin pairs.
            f1_mesh, f2_mesh = np.meshgrid(fA_vec, fB_vec, indexing='ij')
            s1_mesh, s2_mesh = np.meshgrid(sA_vec, sB_vec, indexing='ij')
            
            # Calculate the effective energy: min(S1, S2).
            energy_mesh = np.minimum(s1_mesh, s2_mesh)
            
            # Calculate the roughness kernel R(f1, f2).
            r_mesh = plomp_levelt_raw(f1_mesh, f2_mesh)
            
            # Sum across the full frequency range.
            dissonance = np.sum(energy_mesh * r_mesh)
            
            D[i, j] = dissonance
            D[j, i] = dissonance  # The matrix is symmetric.

    print(f"Calculation complete in {time.time() - start_time:.2f} seconds.")
    return notes, D


# ==========================================
# 4. Plot the heatmap
# ==========================================
def plot_dissonance_heatmap(notes, D, sf2_path):
    plt.figure(figsize=(10, 8))
    
    # Convert MIDI numbers to note names (for example, 60 -> C4).
    note_names = []
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    for n in notes:
        name = f"{names[n % 12]}{n // 12 - 1}"
        note_names.append(name)
        
    # Plot the heatmap with the origin in the lower-left corner.
    im = plt.imshow(D, cmap='magma', origin='lower', aspect='auto')
    
    plt.colorbar(im, label='Sensory Dissonance $D(f_A, f_B)$')
    
    # Set axis labels, showing every few notes.
    step = 3 if len(notes) > 30 else 1
    plt.xticks(ticks=np.arange(0, len(notes), step), labels=note_names[::step], rotation=45)
    plt.yticks(ticks=np.arange(0, len(notes), step), labels=note_names[::step])
    
    plt.xlabel('Base Note A ($f_A$)')
    plt.ylabel('Base Note B ($f_B$)')
    plt.title('SF2 Sensory Dissonance Heatmap $D(f_A, f_B)$')
    
    plt.tight_layout()
    
    # Save to the output directory using the SF2 filename.
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    sf2_name = Path(sf2_path).stem
    output_path = output_dir / f"{sf2_name}_Dissonance_Heatmap.png"
    plt.savefig(output_path, dpi=150)

    # Headless environments use the non-interactive Agg backend.
    if plt.get_backend().lower() == "agg":
        print(f"Heatmap saved to: {output_path}")
    else:
        plt.show()

# ==========================================
# Program entry point
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate an SF2 instrument's dissonance heatmap"
    )
    parser.add_argument(
        "--sf2",
        default=None,
        help="Path to the SF2 file",
    )
    parser.add_argument(
        "--midi-start",
        type=int,
        default=None,
        help="Starting MIDI note (inclusive; defaults to the SF2 keyRange)",
    )
    parser.add_argument(
        "--midi-end",
        type=int,
        default=None,
        help="Ending MIDI note (inclusive; defaults to the SF2 keyRange)",
    )
    args = parser.parse_args()

    if args.midi_start is not None and not 0 <= args.midi_start <= 127:
        parser.error("MIDI notes must be between 0 and 127")
    if args.midi_end is not None and not 0 <= args.midi_end <= 127:
        parser.error("MIDI notes must be between 0 and 127")
    if (
        args.midi_start is not None
        and args.midi_end is not None
        and args.midi_start > args.midi_end
    ):
        parser.error("--midi-start cannot be greater than --midi-end")
    
    try:
        extractor = SF2SpectrumExtractor(args.sf2)
        midi_start = (
            args.midi_start
            if args.midi_start is not None
            else extractor.midi_range.start
        )
        midi_end = (
            args.midi_end
            if args.midi_end is not None
            else extractor.midi_range.stop
        )
        if midi_start > midi_end:
            parser.error("--midi-start cannot be greater than --midi-end")
        notes, D_matrix = calculate_dissonance_matrix(
            extractor,
            midi_range=range(
                midi_start,
                midi_end + 1,
            ),
        )
        plot_dissonance_heatmap(notes, D_matrix, args.sf2)
    except Exception as e:
        print(f"Runtime error (possibly a missing SF2 file or FluidSynth): {e}")