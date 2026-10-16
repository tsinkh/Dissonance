# Dissonance

This project calculates the perceptual dissonance between MIDI notes using the
spectrum of an SF2 SoundFont instrument and the Plomp-Levelt model. The result
is visualized as a heatmap.

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
python main.py --sf2 soundfonts/040_Florestan_String_Quartet.sf2
```

If no MIDI range is specified, the program uses the `keyRange` declared by the
selected SoundFont preset. You can also specify a custom range; the end note
is inclusive:

```bash
python main.py \
  --sf2 soundfonts/040_Florestan_String_Quartet.sf2 \
  --midi-start 55 \
  --midi-end 100
```

MIDI note numbers must be between `0` and `127`. For example:

- Violin: approximately `G3–E7`, or `55–100`
- Piano: approximately `A0–C8`, or `21–108`

Show all available options:

```bash
python main.py --help
```

## Output

Results are saved in the project's `output/` directory. The output filename is
derived from the SF2 filename, for example:

```text
output/040_Florestan_String_Quartet_Dissonance_Heatmap.png
```

The horizontal and vertical axes represent MIDI notes. The color indicates the
dissonance of each note pair.

## Notes

- An SF2 `keyRange` describes the sample-mapping range and may not equal the
  instrument's natural playing range.
- Different instruments have different spectral structures, so absolute
  dissonance values should not be compared directly across SF2 files.
- The program prefers Bank 0 / Program 0. If that preset does not exist, it
  automatically selects the first available preset.
