# About Stable Diffusion model in general
Stable Diffusion is a deep learning, text-to-image model released in 2022. It is primarily used to generate detailed images conditioned on text descriptions, though it can also be applied to other tasks such as inpainting, outpainting, and generating image-to-image translations guided by a text prompt. Stable Diffusion uses a variant of diffusion model (DM), called latent diffusion model (LDM). Stable Diffusion was trained on pairs of images and captions taken from LAION-5B, a publicly available dataset derived from Common Crawl data scraped from the web, where 5 billion image-text pairs were classified based on language, filtered into separate datasets by resolution, a predicted likelihood of containing a watermark, and predicted "aesthetic" score (e.g. subjective visual quality).

# Textual Inversion
**Short description**: How good is textual inversion trained Stable Diffusion 2 at generating new realistic Cézanne landscape paintings? Does a fine-tuned Convolutional Neural Network classify their style as Original, Replica, Stable Diffusion image w/o textual inversion, or General Impressionist Landscape Painting?

**Used data**: The dataset to fine-tune the CNN consists of a range of landscape paintings similar in style to that of Paul Cézanne. The photographs of the images are unfortunately not consistent in quality and were taken from a range of different sources. The images were curated and divided into 5 categories from most (0) to least similar (4) in style.

**Implementation**: The code to generate images with Stable Diffusion 1.5 can be found [here](Textual_Inversion_Metric/StableDiffusion1.5_image_generator.ipynb).

- 0 - authentic Cézanne landscape paintings - 89 images
- 1 - hand-painted replicas and forgeries of authentic Cézanne landscape paintings - 68 images
- 2 - Stable Diffusion 1.5 generated Cézanne landscape paintings with guidance scale 8 (w/o textual inversion) - 88 images
- 3 - Stable Diffusion 1.5 generated Cézanne landscape paintings with guidance scale 0-1 (w/o textual inversion) - 93 images
- 4 - Impressionist landscape paintings from WikiArt dataset - 94 images

Textual Inversion was not implemented when creating the dataset. A separate test dataset with Textual Inversion generated Cézanne landscape paintings was created [here](Textual_Inversion_Metric/StableDiffusion2_textual_inversion_image_generator.ipynb).

The code for Textual Inversion training of Stable Diffusion can be found [here](Textual_Inversion_Metric/StableDiffusion2_textual_inversion_training.ipynb)

- TISD1 - images created by Stable Diffusion 2 with textual inversion trained prompt: "painting in the style of <Cézanne>"
- TISD2 - images created by Stable Diffusion 2 with textual inversion trained prompt: "landscape painting in the style of <Cézanne>"
- TISD3 - images created by Stable Diffusion 2 with textual inversion trained prompt: "painting of the Provence in the style of <Cézanne>"
- TISD4 - images created by Stable Diffusion 2 with textual inversion trained prompt: "painting of Mont Saint Victoire in the style of <Cézanne>"

Before training the CNN, the images were all resized to (512, 512, 3). This was done to disabuse the CNN of learning image sizes. Furthermore, the images were all converted to grayscale. This was done to disabuse the CNN of learning color schemes, as color is difficult to grasp consistently for cameras and depends on lighting etc.. The code for data preparation can be found [here](Textual_Inversion_Metric/Data%20Cleaning.ipynb).

**Evaluation metrics**: Three different fine-tuned Convolutional Neural Network models were employed to judge the quality of the textual inversion generated images: [MobileNet](Textual_Inversion_Metric/Cezanne_MobileNet.ipynb), [EfficinetNetB7](Textual_Inversion_Metric/Cezanne_efficientnetb7.ipynb) and [EfficientNetV2L](Textual_Inversion_Metric/Cezanne_efficientnetv2l.ipynb). All were trained using the Keras libary.
   
**Sources**:
- https://arxiv.org/abs/2208.01618
- https://huggingface.co/docs/diffusers/training/text_inversion
- https://gitlab.com/juliensimon/huggingface-demos/-/blob/main/food102/Stable%20diffusion%20example.ipynb
- https://towardsdatascience.com/deep-image-quality-assessment-30ad71641fac
- https://www.sciencedirect.com/science/article/abs/pii/S0957417418304421
