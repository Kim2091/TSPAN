"""Package the VapourSynth example with its Vapourkit filter. No models/engines."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    root = Path(__file__).resolve().parent
    files = ["tensorrt/vapoursynth_script.vpy", "tensorrt/README.md",
             "vapourkit/TSPAN TensorRT.vkfilter", "vapourkit/README.md", "LICENSE"]
    for name in files:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    output = root / "dist/TSPAN-vapoursynth.zip"
    output.parent.mkdir(exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(root / name, name)
    print(output)


if __name__ == "__main__":
    main()
