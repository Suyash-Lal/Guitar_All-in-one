# Dataset_Scrape.py  (v4 – progress bar + robust)
# ------------------------------------------------
import pathlib, random, shutil, logging, warnings, sys
import pretty_midi
from tqdm import tqdm

SRC_ROOT  = pathlib.Path(r"lmd_full")        # <- adjust if needed
DEST_ROOT = pathlib.Path(r"Dataset")         # <- adjust if needed
FILES_PER = 120                              # per hex folder
SEED      = 42

# General‑MIDI program numbers for guitars (24‑31) + 0 for mis‑labelled tracks
GUITAR_PGMS = set(range(24, 32)) | {0}

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s: %(message)s")
warnings.filterwarnings("ignore",
    message="Tempo, Key or Time signature change events.*")

random.seed(SEED)
DEST_ROOT.mkdir(parents=True, exist_ok=True)

def is_guitar(midi_path: pathlib.Path) -> bool:
    """True if any non‑drum track looks like a guitar."""
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        logging.debug(f"Corrupt {midi_path.name}: {e}")
        return False

    for inst in pm.instruments:
        if inst.is_drum:
            continue
        if inst.program in GUITAR_PGMS:
            return True
        if "guitar" in inst.name.lower():
            return True
    return False


def main() -> None:
    total_copied = 0
    try:
        for hex_dir in sorted(d for d in SRC_ROOT.iterdir() if d.is_dir()):
            mids = list(hex_dir.glob("*.mid"))
            random.shuffle(mids)

            guitar_midis = []
            # tqdm gives a neat per‑folder progress bar
            for f in tqdm(mids, desc=f"Scanning {hex_dir.name}", leave=False):
                if len(guitar_midis) >= FILES_PER:
                    break
                if is_guitar(f):
                    guitar_midis.append(f)

            for f in guitar_midis:
                dst = DEST_ROOT / f"{hex_dir.name}_{f.name}"
                shutil.copy2(str(f), str(dst))
            total_copied += len(guitar_midis)

            logging.info(
                f"{hex_dir.name}: kept {len(guitar_midis):3d} "
                f"(running total {total_copied})"
            )

        logging.info(f"\nFinished. Total guitar MIDIs copied: {total_copied}")

    except KeyboardInterrupt:
        logging.warning(
            f"\nInterrupted by user. Files copied so far: {total_copied}")
        sys.exit(1)


if __name__ == "__main__":
    main()
