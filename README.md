# `match-histogram` Node for InvokeAI
Discord Link :- [match-histogram](https://discord.com/channels/1020123559063990373/1178755835162267658)

An InvokeAI node to match a histogram from one image to another.  This is a bit like the `color correct` node in the main InvokeAI but this works in the YCbCr colourspace and can handle images of different sizes. Also does not require a mask input.
- Option to only transfer luminance channel.
- Option to save output as grayscale

A good use case is to normalize the colors of an image that has been through the tiled scaling workflow of my XYGrid Nodes. 

## Usage
### <ins>Install</ins><BR>
There are two options to install the nodes:

1. **Recommended**: Use Git to clone the repo into the `invokeai/nodes` directory. This allows updating via `git pull`.

    - In the InvokeAI nodes folder, run:

        For Latest Invoke (4.0+):
        ```bash
        git clone https://github.com/skunkworxdark/match_histogram.git
        ```
        For Invoke (3.5-3.7):
        ```bash
        git clone https://github.com/skunkworxdark/match_histogram/tree/invoke-3.7
        ```

2. Manually download and place [match_histogram.py](match_histogram.py) & [__init__.py](__init__.py) in a subfolder in the `invokeai/nodes` folder.

### <ins>Update</ins><BR>
Run a `git pull` from the `match_histogram` folder.

Or run the `update.bat` or `update.sh`.

For manual installs, download and replace the files.

### <ins>Remove</ins><BR>
Delete the `match_histogram` folder or rename it to `_match_histogram` and Invoke will ignore it.

![image](https://github.com/skunkworxdark/match_histogram/assets/21961335/ed12f329-a0ef-444a-9bae-129ed60d6097)
