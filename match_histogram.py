# 2023 skunkworxdark (https://github.com/skunkworxdark)

from typing import Optional

import numpy as np
from PIL import Image

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    FieldDescriptions,
    Input,
    InputField,
    InvocationContext,
    WithMetadata,
    WithWorkflow,
    invocation,
)
from invokeai.app.invocations.primitives import (
    BoardField,
    ImageField,
    ImageOutput,
)
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin


def hist_match(source, template):
    source_shape = source.shape
    # Flatten the source and template images
    source = source.ravel()
    template = template.ravel()

    # Get the set of unique pixel values and their corresponding indices and counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # Calculate the cumulative distribution function (CDF) of the source and template images
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]  # Normalize the CDF of the source image
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]  # Normalize the CDF of the template image

    # Interpolate pixel values of the source image based on matching CDFs
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values).astype(int)

    return interp_t_values[bin_idx].reshape(source_shape)


@invocation(
    "match_histogram",
    title="Match Histogram",
    tags=["histogram", "color", "image"],
    category="color",
    version="1.0.0",
)
class MatchHistogramInvocation(BaseInvocation, WithWorkflow, WithMetadata):
    """match a histogram from one image to another"""

    # Inputs
    board: Optional[BoardField] = InputField(default=None, description=FieldDescriptions.board, input=Input.Direct)
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
        source = context.services.images.get_pil_image(self.image.image_name)
        reference = context.services.images.get_pil_image(self.reference_image.image_name)

        source_has_alpha = source.mode == 'RGBA'
        if source_has_alpha:
            source_alpha = source.split()[3]

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
        matched_channels = []
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

        if source_has_alpha:
            output_image.putalpha(source_alpha)

        # Save the image
        image_dto = context.services.images.create(
            image=output_image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            board_id=self.board.board_id if self.board else None,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            metadata=self.metadata,
            workflow=self.workflow,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )
