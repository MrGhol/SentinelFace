Model Files (Not in Git)

SentinelFace expects the following files to be present locally:

- `models/buffalo_l/det_10g.onnx`
- `models/buffalo_l/w600k_r50.onnx`
- `models/gender.onnx`
- `models/age.onnx`

InsightFace (SCRFD + ArcFace)

Download the InsightFace model zoo pack `buffalo_l.zip` and extract it into `models/buffalo_l/`.
The pack includes `det_10g.onnx` (SCRFD) and `w600k_r50.onnx` (ArcFace), among other files.

Sources:
- InsightFace model zoo mirror (buffalo_l.zip): https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download

FairFace (Gender/Age)

The official FairFace project provides pretrained weights (PyTorch). This project expects
ONNX exports for gender and age, saved as `models/gender.onnx` and `models/age.onnx`.

Sources:
- FairFace repository: https://github.com/dchen236/FairFace
- Pretrained weights (Google Drive link from the FairFace README): https://drive.google.com/drive/folders/1F_pXfbzWvG-bhCpNsRj6F_xsdjpesiFu?usp=sharing

Note:
If you export from the FairFace weights, ensure the ONNX outputs correspond to the
gender and age classifiers used by your pipeline.
