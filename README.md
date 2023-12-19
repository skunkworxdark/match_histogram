# match-histogram-node
An InvokeAI node to match a histogram from one image to another.  This is a bit like the `color correct` node in the main InvokeAI but this works in the YCbCr colourspace and can handle images of different sizes. Also does not require a mask input.
- Option to only transfer luminance channel.
- Option to save output as grayscale

A good use case is to normalize the colors of an image that has been through the tiled scaling workflow of my XYGrid Nodes. 

## Usage
<ins>Install:</ins><BR>
There are two options for installing these nodes. (Option 1 is the recommended option) 
1. Git clone the repo into the `invokeai/nodes` directory. (**Recommended** - as it allows updating via a git pull)
    - open a command prompt/terminal in the invokeAI nodes directory ( or choose `8. Open the developer console` option from the invoke.bat then `cd nodes`)
    - run `git clone https://github.com/skunkworxdark/match_histogram.git`
2. Manually download and place [match_histogram.py](match_histogram.py) & [__init__.py](__init__.py) in a subfolder in the `invokeai/nodes` folder.

<ins>Update:</ins><BR>
Run a `git pull` from the `match_histogram` folder. Or run the `update.bat` or `update.sh` that is in the `invokeai/nodes/match_histogram` folder. If you installed it manually then the only option is to monitor the repo or discord channel and manually download and replace the file yourself.

<ins>Remove:</ins><BR>
Simply delete the `match_histogram` folder or you can rename it by adding an underscore `_match_histogram` and Invoke will ignore it.

![image](https://github.com/skunkworxdark/match_histogram/assets/21961335/ed12f329-a0ef-444a-9bae-129ed60d6097)
