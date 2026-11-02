# Exporting browser bots

Website visitors need only a browser. Python and these optional dependencies are
needed by the developer producing model assets:

```sh
.venv/bin/python -m pip install -r training/requirements-web.txt
.venv/bin/python training/export_web.py path/to/model.pt \
  --id my-bot --name 'My bot' --simulations 32
```

Only export trusted checkpoints: the existing AZ checkpoint loader uses pickle.
The exporter supports AZ ResNets with 19 or 10 input planes, and retains dropped
plane masks. It creates a fixed float32 `[1,19,9,9]` input named `features`; compact
plane selection happens inside the exported graph. Outputs are `logits` with
shape `[1,209]` and `value` with shape `[1]`, both in the side-to-move perspective.

The command verifies ONNX outputs against PyTorch on sixteen deterministic native
engine positions before publishing assets. `public/models/` receives the ONNX
model with its content hash in its filename, a provenance/parity report, and
reference fixtures with native features, legal moves, canonical moves and neural
outputs. `public/bots.json` is updated by stable bot ID, preserving other bots.

The fixtures are also intended for browser verification: compare every feature
plane and policy remapping, then compare inference using a small floating-point
tolerance. Native ONNX CPU parity does not by itself establish browser execution
or whole-search parity. After replacing a checkpoint, repeat browser inference
and complete-move checks before release.

The checked-in initial catalog contains a 64-channel, six-block pretrained model
and a 128-channel, ten-block self-play model. Their names describe provenance,
not a measured strength ranking. Source checkpoint SHA-256 checksums are recorded
in the model reports; experiment files themselves remain local and ignored.

Run the export regression test with:

```sh
.venv/bin/python -m unittest discover -s training -p test_export_web.py -v
```

Deploy `public` assets through the normal Vite production build. Model URLs in
the manifest are relative to the app base. Serve versioned ONNX assets with long
cache lifetimes, while allowing the manifest to refresh. Weight files are public
downloads once deployed.
