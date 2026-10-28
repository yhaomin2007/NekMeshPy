# NekMeshPy examples

This staging tree contains two reproducible all-hexahedral mesh examples:

- `aircraft_farfield/`: a full-aircraft near-body and spherical far-field mesh.
- `pipe_network_features/`: a connected multi-diameter pipe network with bends,
  a wall protrusion, and an internal baffle/obstacle.

Each example includes source scripts, compact inputs, mesh quality checks, a
preview, and a gzip-compressed Nek5000/NekRS `.re2` mesh. The larger aircraft
mesh is stored in numbered parts; its README gives the lossless reassembly
command. The smaller pipe mesh can be decompressed directly with
`gzip -dk output/*.re2.gz`.

GitHub transport chunks are capped below 1 MB. After cloning, first run:

```bash
python examples/restore_large_files.py
```

This reconstructs every split binary input, preview, and compressed mesh.

Generated visualization/field files and optimization checkpoints are omitted
because they can be recreated from the included scripts and are much larger
than the source inputs.
