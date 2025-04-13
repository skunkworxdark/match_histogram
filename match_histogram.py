# 2024 skunkworxdark (https://github.com/skunkworxdark)
# Updated 2025-04-13 to use OpenCV instead of scikit-image for LAB conversion

from typing import Any

import cv2
import numpy as np
from PIL import Image

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
    """Matches the histogram of the source image to the template image."""
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
    title="Match Histogram (YCbCr)",
    tags=["histogram", "color", "image"],
    category="color",
    version="1.1.2",
)
class MatchHistogramInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Match a histogram from one image to another using YCbCr color space"""

    # Inputs
    image: ImageField = InputField(description="The image to receive the histogram")
    reference_image: ImageField = InputField(description="The reference image with the source histogram")
    match_luminance_only: bool = InputField(
        default=False,
        description="Only transfer the luminance",
    )
    output_grayscale: bool = InputField(
        default=False,
        description="Convert output image to grayscale",
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        source = context.images.get_pil(self.image.image_name)
        reference = context.images.get_pil(self.reference_image.image_name)

        # Extract alpha channel if present
        source_alpha = source.split()[-1] if source.mode == "RGBA" else None

        # Check if the source and reference images are colored
        source_is_rgb = source.mode == "RGB" or source.mode == "RGBA"
        reference_is_rgb = reference.mode == "RGB" or reference.mode == "RGBA"

        # Convert the source and template images to 'YCbCr' if they are colored, 'L' otherwise
        source_yuv = source.convert("YCbCr") if source_is_rgb else source.convert("L")
        reference_yuv = reference.convert("YCbCr") if reference_is_rgb else reference.convert("L")

        # Split the source and template images into their respective channels
        source_channels_pil = list(source_yuv.split())
        reference_channels_pil = list(reference_yuv.split())

        # Match the histograms of the source and template images
        matched_channels: list[Image.Image] = []
        num_source_channels = len(source_channels_pil)
        num_ref_channels = len(reference_channels_pil)

        for i in range(num_source_channels):
            # If matching only the luminance channel or the template image is grayscale,
            # leave the chrominance channels unchanged (if they exist)
            if (self.match_luminance_only or not reference_is_rgb) and source_is_rgb and i > 0:
                matched_channels.append(source_channels_pil[i])
            else:
                # Ensure reference channel exists (handles color -> grayscale matching)
                ref_channel_idx = min(i, num_ref_channels - 1)

                # Convert PIL channels to NumPy arrays for hist_match
                source_arr = np.array(source_channels_pil[i])
                ref_arr = np.array(reference_channels_pil[ref_channel_idx])

                # Match the histogram
                matched_arr = hist_match(source_arr, ref_arr)

                # Convert back to PIL Image (ensure uint8 for L mode)
                matched_channel_pil = Image.fromarray(matched_arr.astype("uint8"), "L")
                matched_channels.append(matched_channel_pil)

        # Merge the matched channels to get the output image
        if source_is_rgb:
            # Ensure we have 3 channels to merge for YCbCr
            while len(matched_channels) < 3:
                # This case should ideally not happen if source_is_rgb is true,
                # but as a fallback, append original Cb/Cr or default gray
                if len(source_channels_pil) > len(matched_channels):
                    matched_channels.append(source_channels_pil[len(matched_channels)])
                else:  # Fallback if original source wasn't 3 channels (e.g., LA)
                    matched_channels.append(Image.new("L", source_channels_pil[0].size, 128))

            output_image = Image.merge("YCbCr", matched_channels[:3]).convert("RGB")  # Use first 3

            if self.output_grayscale:
                output_image = output_image.convert("L")
        else:  # Source was Grayscale ('L')
            output_image = matched_channels[0]

        # Restore alpha channel if present in source
        if source_alpha is not None:
            if output_image.mode == "L":
                output_image = output_image.convert("LA")  # Add alpha
                output_image.putalpha(source_alpha)
            elif output_image.mode == "RGB":
                output_image = output_image.convert("RGBA")  # Add alpha
                output_image.putalpha(source_alpha)
            # else leave as is (e.g. if already RGBA - though unlikely here)

        # Save the image
        image_dto = context.images.save(output_image)

        return ImageOutput.build(image_dto)


@invocation(
    "match_histogram_lab",
    title="Match Histogram LAB",
    tags=["histogram", "color", "image"],
    category="color",
    version="1.1.0",
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
        source_alpha = source.split()[-1] if source.mode == "RGBA" else None

        # Always convert to RGB first for color processing or reference
        source_rgb = source.convert("RGB")
        reference_rgb = reference.convert("RGB")  # Also convert reference for consistency

        # Convert to numpy arrays (uint8, 0-255)
        source_np_rgb = np.array(source_rgb)
        reference_np_rgb = np.array(reference_rgb)

        if source_is_luminance_only:
            # Special handling if the original source image was grayscale
            # We match its L channel to the L channel of the reference's LAB version

            # Convert reference to LAB
            ref_np_float = reference_np_rgb.astype(np.float32) / 255.0
            ref_lab = cv2.cvtColor(ref_np_float, cv2.COLOR_RGB2Lab)

            # Get source L channel (already grayscale) and reference L* channel
            source_l_channel = source_np_rgb[:, :, 0]  # It's RGB but all channels are the same
            ref_l_channel = ref_lab[:, :, 0]

            # Match histograms for the L channel
            matched_l_channel = hist_match(source_l_channel, ref_l_channel)

            # Output is grayscale
            output_array = matched_l_channel.astype(np.uint8)
            output_image = Image.fromarray(output_array, mode="L")

        else:
            # Regular color processing (source was RGB or RGBA)

            # Convert source and reference RGB to LAB (float32)
            # OpenCV expects float32 input in range [0, 1] for RGB->Lab
            source_np_float = source_np_rgb.astype(np.float32) / 255.0
            reference_np_float = reference_np_rgb.astype(np.float32) / 255.0

            source_lab = cv2.cvtColor(source_np_float, cv2.COLOR_RGB2Lab)
            reference_lab = cv2.cvtColor(reference_np_float, cv2.COLOR_RGB2Lab)

            # Match histograms channel by channel
            # OpenCV LAB channels: L* (0-100), a* (-127-127), b* (-127-127) approx
            matched_lab = np.copy(source_lab)  # Start with a copy of source LAB

            # Match L* channel
            matched_lab[:, :, 0] = hist_match(source_lab[:, :, 0], reference_lab[:, :, 0])

            # Match a* and b* channels if not luminance only
            if not self.match_luminance_only:
                matched_lab[:, :, 1] = hist_match(source_lab[:, :, 1], reference_lab[:, :, 1])
                matched_lab[:, :, 2] = hist_match(source_lab[:, :, 2], reference_lab[:, :, 2])

            # Convert back to RGB
            # cv2.COLOR_Lab2RGB expects float32 Lab input
            matched_rgb_float = cv2.cvtColor(matched_lab.astype(np.float32), cv2.COLOR_Lab2RGB)

            # Clip values to [0, 1] range as conversion can sometimes slightly exceed bounds
            matched_rgb_float_clipped = np.clip(matched_rgb_float, 0, 1)

            # Convert back to 8-bit RGB (0-255)
            matched_rgb_8bit = (matched_rgb_float_clipped * 255).astype(np.uint8)
            output_image = Image.fromarray(matched_rgb_8bit, mode="RGB")

            # Convert to grayscale if requested
            if self.output_grayscale:
                output_image = output_image.convert("L")

        # Restore alpha channel if present in source
        if source_alpha is not None:
            # Ensure output mode supports alpha before adding it
            if output_image.mode == "L":
                output_image = output_image.convert("LA")  # Add alpha
                output_image.putalpha(source_alpha)
            elif output_image.mode == "RGB":
                output_image = output_image.convert("RGBA")  # Add alpha
                output_image.putalpha(source_alpha)
            # else leave as is (e.g. if already RGBA/LA)

        # Save the image
        image_dto = context.images.save(output_image)

        return ImageOutput.build(image_dto)
