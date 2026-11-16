# Learning notes


The original diagrams, equations and literature comparison below are background notes; they are not project result figures or evidence that every listed method was evaluated.

- Wan2.1

    ![Wan framework](../assets/Wan_Fig9.png)

    - VAE

        ![Wan VAE](../assets/Wan_Fig8.png)

    - DiT

        ![Wan DiT](../assets/Wan_Fig10.png)

    - Diffusion


        |  | training | sampling |
        | --- | --- | --- |
        | input | $z_t=(1-t)z_0+t\epsilon$ | $z_1=\epsilon$ 让 $t$ 递减 |
        | Step 1 | $\hat v_t=f_\theta(z_t,t,c)$ | $\hat v_t=f_\theta(z_t,t,c)$ Adaptive Caching |
        | Step 2 | $\mathcal L=\lVert\hat v-(\epsilon-z_0)\rVert^2$ | $z_{t-T}=\operatorname{Sampler}(z_t,\hat v_t,t,t-T)$ |
        | output | 训练好的 $\theta$ | 生成视频 $\hat z_0$ |
- Adaptive Caching → Efficient Generation
    - skip/reuse or compute/refresh?

    |  |  | 任务 |  | backbone | dataset | SeaCache | SenCache |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | SeaCache | CVPR 2026 Oral | T2I<br>T2V | when | FLUX.1-dev<br>**Wan2.1**-1.3B<br>HunyuanVideo | DrawBench<br>**VBench** |  |  |
    | SenCache | CVPR 2026 Oral | T2V | when | **Wan2.1**<br>CogVideoX<br>LTX-Video | **VBench**<br>T2V-CompBench |  |  |
    | Uncached |  |  |  |  |  |  |  |
    | TeaCache | CVPR 2025 Highlight | T2V | when | Open-Sora<br>Latte | **VBench**<br>T2V-CompBench | 主表 | 主表 |
    | MagCache | NeurIPS 2025 | T2I<br>T2V | when | FLUX<br>**Wan2.1**-1.3B<br>Open-Sora 1.2<br>HunyuanVideo<br>CogVideoX | **VBench** | 附录 | 主表 |
    | TaylorSeer | ICCV 2025 | T2I<br>T2V<br>C2I | how | FLUX.1-dev<br>HunyuanVideo<br>DiT-XL/2<br>HiDream<br>FramePack<br>**Wan2.1** | DrawBench<br>**VBench**<br>ImageNet-256 | 主表 | / |
    | Δ-DiT | IJCV 2026 | T2I<br>C2I | which | PixArt-α<br>DiT-XL<br>SD3 | ImageNet<br>MS-COCO2017<br>PartiPrompts | Table 1 | / |
    | ToCa | ICLR 2025 | T2I<br>T2V<br>C2I | which | PixArt-α<br>OpenSora<br>DiT-XL/2<br>FLUX | ImageNet-256<br>**VBench**<br>MS-COCO2017<br>PartiPrompts | Table 1 | / |
    | DiCache | ICLR 2026 | T2I<br>T2V | when<br>how | FLUX.1-dev<br>**Wan2.1**-1.3B<br>HunyuanVideo | LAION-5B<br>VideoDPO+VideoFeedback 经 GPT-4o 润色 | 消融 | / |
    | D2Cache | CVPR 2026 | T2V | how | **Wan2.1**-1.3B<br>Open-Sora 1.2<br>LTX-Video<br>Latte | **VBench** | / | / |
    | TC-Padé | CVPR 2026 | T2I<br>T2V<br>C2I | when<br>how | FLUX.1-dev<br>**Wan2.1**-1.3B<br>DiT-XL/2 | ImageNet-256<br>**VBench**-2.0<br>COCO2017 | / | / |
    | BWCache | ICLR 2026 | T2V | when<br>which | **Wan2.1**<br>Open-Sora<br>Latte<br>HunyuanVideo | Open-Sora<br>**VBench**<br>T2V-CompBench | / | / |
    | SVD-Cache | CVPR 2026 | T2I<br>T2V | how | FLUX.1-dev<br>HunyuanVideo | DrawBench<br>**VBench** | / | / |
    | WorldCache | ECCV 2026 | T2W<br>I2W | when<br>how | **Wan2.1**<br>Cosmos-Predict2.5 | PAI-Bench<br>EgoDex-Eval | / | / |
- SeaCache — CVPR 2026 Oral

    ![SeaCache Figure 3](../assets/SeaCache_Fig3.png)

- SenCache — CVPR 2026 Oral

    ![SenCache Figure 2](../assets/SenCache_Fig2.png)


![SeaCache and SenCache comparison](../assets/Sea_Sen_Comparison.png)
