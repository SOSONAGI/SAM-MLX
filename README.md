# SAM-MLX
one-click segment anything via gradio app

You can attach image and segment anything from the SAM-VIT model.

How to

1. Clone repository from the mlx and mlx-examples

1-1 clone the mlx
```CLI
git clone https://github.com/ml-explore/mlx.git
```

1-2 clone the mlx-examples
```CLI
git clone https://github.com/ml-explore/mlx-examples.git
```

then go to segment_anything path 

2. Install the requirements

```CLI
pip install mlx==0.15.0
pip install mlx_lm
pip install gradio
pip install opencv-python
```

3. You have to download app.py into segment_anything path

4. Download the model or convert the mlx format from the sam-vit-base or other type model

4-1 Download link
```CLI
git clone https://huggingface.co/sosoai/sam-vit-base-mlx
```

or refer to below mlx_lm links for your own model convert

https://github.com/ml-explore/mlx-examples/tree/main/segment_anything


This still needs lots of things for development in that output of segment is not correct as notebooks.

Really hope for contribution.

TO DO
1. Make the app in gradio (V)
2. share in github (V)
3. improve the performance for segment and box

Licenses
Followed everythings from the SAM model (Apache 2.0) and MLX.

