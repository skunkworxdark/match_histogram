# 2024 skunkworxdark (https://github.com/skunkworxdark)

from typing import Any

import numpy as np
from PIL import Image
from skimage import color as skcolor

from invokeai.invocation_api import (
    BaseInvocation,
    ImageField,
    ImageOutput,
    InputField,
    InvocationContext,
    WithBoard,
    WithMetadata,
    invocation,
)


def hist_match(source: np.ndarray[Any, Any], template: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    # Flatten the source and template images
    source_flat = source.ravel()
    template_flat = template.ravel()

    # Get the set of unique pixel values and their corresponding indices and counts
    _, bin_idx, s_counts = np.unique(source_flat, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template_flat, return_counts=True)

    # Calculate the cumulative distribution function (CDF) of the source and template images
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]  # Normalize the CDF of the source image
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]  # Normalize the CDF of the template image

    # Interpolate pixel values of the source image based on matching CDFs
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    # Create the mapping array and reshape to original dimensions
    mapped = interp_t_values[bin_idx].reshape(source.shape)

    return mapped.astype(source.dtype)


@invocation(
    "match_histogram",
    title="Match Histogram",
    tags=["histogram", "color", "image"],
    category="color",
    version="1.1.1",
)
class MatchHistogramInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Match a histogram from one image to another using YCbCr color space"""

    # Inputs
    image: ImageField = InputField(description="The image to receive the histogram")
    reference_image: ImageField = InputField(description="The reference image with the source histogram")
    match_luminance_only: bool = InputField(
        default=False,
        description="only transfer the luminance",
    )
    output_grayscale: bool = InputField(
        default=False,
        description="convert output image to grayscale",
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        source = context.images.get_pil(self.image.image_name)
        reference = context.images.get_pil(self.reference_image.image_name)

        # Extract alpha channel if present
        source_alpha = source.split()[3] if source.mode == "RGBA" else None

        # Check if the source and reference images are colored
        source_is_rgb = source.mode == "RGB" or source.mode == "RGBA"
        reference_is_rgb = reference.mode == "RGB" or reference.mode == "RGBA"

        # Convert the source and template images to 'YCbCr' if they are colored, 'L' otherwise
        source_yuv = source.convert("YCbCr") if source_is_rgb else source.convert("L")
        reference_yuv = reference.convert("YCbCr") if reference_is_rgb else reference.convert("L")

        # Split the source and template images into their respective channels
        source_channels = source_yuv.split()
        reference_channels = reference_yuv.split()

        # Match the histograms of the source and template images
        matched_channels: list[Image.Image] = []
        for i in range(len(source_channels)):
            # If matching only the luminance channel or the template image is grayscale, leave the chrominance channels unchanged
            if (self.match_luminance_only or not reference_is_rgb) and source_is_rgb and i > 0:
                matched_channels.append(source_channels[i])
            else:
                # Match the histogram of the current channel of the source image to that of the template image
                matched_channel = Image.fromarray(
                    hist_match(np.array(source_channels[i]), np.array(reference_channels[i])).astype("uint8"), "L"
                )
                matched_channels.append(matched_channel)

        # Merge the matched channels to get the output image
        if source_is_rgb:
            output_image = Image.merge("YCbCr", matched_channels).convert("RGB")
            if self.output_grayscale:
                output_image = output_image.convert("L")
        else:
            output_image = matched_channels[0]

        if source_alpha is not None:
            output_image.putalpha(source_alpha)

        # Save the image
        image_dto = context.images.save(output_image)

        return ImageOutput.build(image_dto)


@invocation(
    "match_histogram_lab",
    title="Match Histogram LAB",
    tags=["histogram", "color", "image"],
    category="color",
    version="1.0.0",
)
class MatchHistogramLabInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Match a histogram from one image to another using Lab color space"""

    # Inputs
    image: ImageField = InputField(description="The image to receive the histogram")
    reference_image: ImageField = InputField(description="The reference image with the source histogram")
    match_luminance_only: bool = InputField(
        default=False,
        description="Only transfer the luminance/brightness channel",
    )
    output_grayscale: bool = InputField(
        default=False,
        description="Convert output image to grayscale",
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        source = context.images.get_pil(self.image.image_name)
        reference = context.images.get_pil(self.reference_image.image_name)

        # Check if source is luminance-only (grayscale)
        source_is_luminance_only = source.mode == "L"

        # Extract alpha channel if present
        source_alpha = source.split()[3] if source.mode == "RGBA" else None

        # Always convert to RGB first (unless source is luminance-only)
        if source_is_luminance_only:
            # For luminance-only source, convert reference to grayscale too
            source_gray = source
            reference_gray = reference.convert("L")

            # Direct luminance-to-luminance histogram matching
            output_array = hist_match(np.array(source_gray), np.array(reference_gray)).astype(np.uint8)

            output_image = Image.fromarray(output_array, mode="L")
        else:
            # Regular color processing
            source_rgb = source.convert("RGB")
            reference_rgb = reference.convert("RGB")

            # Convert to numpy arrays
            source_np = np.array(source_rgb)
            reference_np = np.array(reference_rgb)

            # Convert to LAB color space
            source_channels = skcolor.rgb2lab(source_np)
            reference_channels = skcolor.rgb2lab(reference_np)

            # Process each channel
            for i in range(3):
                # Only process first channel (luminance) or if matching all channels
                if i == 0 or not self.match_luminance_only:
                    source_channels[:, :, i] = hist_match(source_channels[:, :, i], reference_channels[:, :, i])

            # Convert back to RGB
            matched_rgb = skcolor.lab2rgb(source_channels)
            np.clip(matched_rgb, 0, 1)

            # Convert to 8-bit and create PIL image
            matched_rgb_8bit = (matched_rgb * 255).astype(np.uint8)
            output_image = Image.fromarray(matched_rgb_8bit, mode="RGB")

            # Convert to grayscale if requested or if source was grayscale
            if self.output_grayscale:
                output_image = output_image.convert("L")

        # Restore alpha channel if present in source
        if source_alpha is not None:
            output_image.putalpha(source_alpha)

        # Save the image
        image_dto = context.images.save(output_image)

        return ImageOutput.build(image_dto)
