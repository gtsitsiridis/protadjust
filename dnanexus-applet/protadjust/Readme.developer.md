# ProtAdjust — Developer Readme

## Implementation

The applet is a thin shell wrapper around the [`ghcr.io/gtsitsiridis/protadjust`](https://github.com/gtsitsiridis/protadjust/pkgs/container/protadjust) Docker image. On each run it:

1. Pulls the image from GHCR.
2. Downloads the required input files from the DNAnexus platform.
3. Runs `protadjust <method> ...` inside the container with the working directory mounted at `/data`.
4. Uploads `output/adjusted_proteomics.parquet` as the output file.

## Building and deploying

```bash
# from the dnanexus-applet/protadjust directory
dx build .
```

To update the image version used at runtime, edit the `IMAGE` variable in `src/protadjust.sh`.

## Updating the applet

After changing `dxapp.json` or `src/protadjust.sh`, rebuild with `dx build .` and re-run your test jobs. The applet pulls the Docker image fresh on every run, so a new image release does not require rebuilding the applet.

## Instance types

The default instance (`mem1_ssd1_v2_x16`) suits most methods. `protrider` is memory-intensive — override via `systemRequirements` at job submission if needed:

```json
{
  "systemRequirements": {
    "main": {"instanceType": "mem2_ssd1_v2_x32"}
  }
}
```

## Entry points

- `main` — single entry point that handles all methods.
