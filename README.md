# Dissonance

This project calculates the perceptual dissonance between MIDI notes using the spectrum of an SF2 SoundFont instrument and the Plomp-Levelt model. The result is visualized as a heatmap.

The program:

1. Loads the specified `.sf2` file.
2. Automatically selects an available SoundFont preset.
3. Reads the preset's declared MIDI key range.
4. Renders notes and extracts their spectra.
5. Calculates the dissonance between every pair of notes.
6. Saves the result as a PNG heatmap.

## Installation

Using a Python virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The system-level FluidSynth library is also required. On Debian/Ubuntu:

```bash
sudo apt install fluidsynth libfluidsynth3
```

## Usage

Run the program with an SF2 file:

```bash
python main.py --sf2 000_000_Piano_1
```

The program automatically looks for a filename without the `.sf2` suffix in the `soundfonts/` directory. The explicit path form is also supported:

```bash
python main.py --sf2 soundfonts/000_000_Piano_1.sf2
```

If no MIDI range is specified, the program uses the `keyRange` declared by the selected SoundFont preset. You can also specify a custom range; the end note is inclusive:

```bash
python main.py \
  --sf2 000_000_Piano_1 \
  --midi-start 21 \
  --midi-end 108
```

MIDI note numbers must be between `0` and `127`. For example:

- Piano: approximately `A0–C8`, or `21–108`
- Violin: approximately `G3–E7`, or `55–100`

To calculate and plot only one row of the dissonance matrix, use `--note` with a note name such as `A3`. The horizontal axis still uses the selected MIDI range:

```bash
python main.py \
  --sf2 000_000_Piano_1 \
  --note A3 \
  --midi-start 21 \
  --midi-end 108
```

This produces a bar chart containing `D(A3, f_B)` for every note in the comparison range.

Show all available options:

```bash
python main.py --help
```

## Output

Results are saved in the project's `output/` directory. The output filename includes the SF2 filename and the actual note range, for example:

```text
output/000_000_Piano_1_A0-C8.png
```

The horizontal and vertical axes represent MIDI notes. The color indicates the dissonance of each note pair.

When `--note` is used, the output is a bar chart instead of a heatmap. The output filename includes the selected note, for example:

```text
output/000_000_Piano_1_A3_A0-C8.png
```

Sharp note names are supported, from `C-1` to `G9`.

## Notes

- An SF2 `keyRange` describes the sample-mapping range and may not equal the instrument's natural playing range.
- Different instruments have different spectral structures, so absolute dissonance values should not be compared directly across SF2 files.
- The program prefers Bank 0 / Program 0. If that preset does not exist, it automatically selects the first available preset.
