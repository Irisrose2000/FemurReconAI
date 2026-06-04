from inference.pipeline import FemurReconstructionPipeline

pipe = FemurReconstructionPipeline.from_checkpoints(
    seg_ckpt="../checkpoints/seg_best.pth",
    comp_ckpt="../checkpoints/comp_best.pth",
    device="cpu"
)

result = pipe.run(
    r"C:\Users\Ann Lia Sunil\Desktop\fem\final\femur_dataset\data\processed\sample_00001.npz"
)

print(result.summary())

result.save("../results/test_run/")