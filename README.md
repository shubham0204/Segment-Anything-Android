# Segment-Anything (SAM) and SAM v2 Inference In Android 

![app_snapshots](https://github.com/user-attachments/assets/863b7774-4c89-4b1d-94b8-7b4edadef6d5)

- On-device inference of SAM/SAM2 with `onnxruntime`
- Clean Kotlin-only implementation, with no additional code compilation 
- No support for text-prompt as an input to the model
- The inference time is *quite high* even with `float16` quantization enabled

## About Segment-Anything

![sam_architecture](https://github.com/user-attachments/assets/6a982571-7366-4849-b716-635786207bae)

- Large-language models have demonstrated significant performance gains in numerous NLP tasks within zero or few-shot problem settings. The prompt or a text given at inference-time to the LLM guides the generation of the output.
- Foundation models like CLIP and ALIGN have been popular due to wide adaptability and fine-tuning capabilities for downstream tasks.
- The goal of the authors is to build a **foundation model for image segmentation**.

#### Task
- Authors define a **promptable image segmentation task**.
- The **prompt** could be **spatial or textual information** which guides the model to generate the desired segmentation mask.

#### Model
- A powerful **image encoder** is used to produce image embeddings and a **prompt encoder** embeds prompts, both of which are combined with a **mask decoder**.
- The authors focus on **point, box and mask prompts** with initial results on free-form text prompts.
- **Image Encoder**: MAE (Masked Autoencoder) pre-trained Vision Transformer
- **Prompt Encoder**: Points and boxes are represented by positional encodings, masks are embedded with convolutional layers, and free-form text with an encoder like CLIP
- **Mask Decoder**: Transformer-based decoder model

#### Data Engine
- To achieve strong generalization on unknown datasets, authors propose a model-in-the-loop data annotation process with three phases.
- In the ***assisted-manual phase***, SAM helps annotators in annotating masks.
- In the ***semi-automatic phase***, SAM automatically generates masks for certain objects, by prompting their locations in the image.
- In the ***fully-automatic phase***, SAM is prompted with a regular grid of foreground points, each of which yields a segmentation mask.

## Setup

1. Clone the project from GitHub and open the resulting directory in Android Studio.

```text
git clone --depth=1 https://github.com/shubham0204/Segment-Anything-Android
```

2. Android Studio starts building the project automatically. If not, select **Build > Rebuild Project** to start a project build.

3. After a successful project build, [connect an Android device](https://developer.android.com/studio/run/device) to your system. Once connected, the name of the device must be visible in top menu-bar in Android Studio.

4. Download any `*_encoder.onnx` and corresponding `*_decoder.onnx` models from the [HuggingFace repository](https://huggingface.co/shubham0204/sam2-onnx-models) and place them in the root directory of the project.

5. Using the `adb` CLI tool, insert the ONNX models in the device's storage,

```text
adb push sam2_hiera_small_encoder.onnx /data/local/tmp/sam/encoder.onnx
adb push sam2_hiera_small_decoder.onnx /data/local/tmp/sam/decoder.onnx
```

Replace `sam2_hiera_small_decoder.onnx` and `sam2_hiera_small_encoder.onnx` with the name of the model downloaded from the HF repository in step (4).

6. Update the model paths and set other options in `MainActivity.kt`,

```kotlin
class MainActivity : ComponentActivity() {

    // ...

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            SAMAndroidTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Column(
                        // ...
                    ) {

                        // ...

                        LaunchedEffect(0) {
                            // ...
                            // The paths below should match the ones
                            // used in step (5)
                            encoder.init(
                                "/data/local/tmp/sam/encoder_fp16.onnx",
                                useXNNPack = true,
                                useFP16 = true
                            )
                            decoder.init(
                                "/data/local/tmp/sam/decoder_fp16.onnx",
                                useXNNPack = true,
                                useFP16 = true
                            )
                            // ...
                        }
                        
                        // ...
                    }
                }
            }
        }
    }
}
```

## Resources

- [ONNX-SAM2-Segment-Anything](https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything): ONNX models were derived from the Colab notebook linked in the `README.md` of this project.
- [Segment Anything - arxiv](https://arxiv.org/abs/2304.02643)
- [SAM 2: Segment Anything in Images and Videos - arxiv](https://arxiv.org/abs/2408.00714)

## Citations

```text
@misc{ravi2024sam2segmentimages,
      title={SAM 2: Segment Anything in Images and Videos}, 
      author={Nikhila Ravi and Valentin Gabeur and Yuan-Ting Hu and Ronghang Hu and Chaitanya Ryali and Tengyu Ma and Haitham Khedr and Roman Rädle and Chloe Rolland and Laura Gustafson and Eric Mintun and Junting Pan and Kalyan Vasudev Alwala and Nicolas Carion and Chao-Yuan Wu and Ross Girshick and Piotr Dollár and Christoph Feichtenhofer},
      year={2024},
      eprint={2408.00714},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.00714}, 
}
```

```text
@misc{kirillov2023segment,
      title={Segment Anything}, 
      author={Alexander Kirillov and Eric Mintun and Nikhila Ravi and Hanzi Mao and Chloe Rolland and Laura Gustafson and Tete Xiao and Spencer Whitehead and Alexander C. Berg and Wan-Yen Lo and Piotr Dollár and Ross Girshick},
      year={2023},
      eprint={2304.02643},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2304.02643}, 
}
```
