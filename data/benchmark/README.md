# Original IDPC-DU benchmark instances

Source: [Mendeley Data v3](https://data.mendeley.com/datasets/t726xwcjf9/3), DOI 10.17632/t726xwcjf9.3 (2020), CC BY 4.0.

All 42 original files are unpacked: 24 in set1 and 18 in set2. They were extracted from the supplied t726xwcjf9-3.zip, which contains IDPC-DU/set1.rar and IDPC-DU/set2.rar. The benchmark files have not been rewritten or preprocessed.

Format: first line N D; second line s t; remaining lines u v weight domain. Original node/domain labels are one-based; graph parsing normalizes node labels and assigns zero-based edge IDs in original file order.

To restore the complete archive:

```powershell
python tools/unpack_idpc.py "C:\Users\FPT SHOP\Downloads\t726xwcjf9-3.zip" --set all
```

The parser checks header ranges, edge fields and filename counts. New experiment manifests record the SHA256 of each input file. See docs/REPRODUCE_PAPER.md for commands and scientific limitations.
